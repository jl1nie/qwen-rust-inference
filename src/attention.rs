use crate::{
    ops::{rms_norm, sigmoid, softmax, Scale},
    state::KvCache,
    tensor::{finite, require, Backend, Matrix, Result},
};
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub theta: f32,
    pub eps: f32,
}
#[derive(Debug, Clone)]
pub struct AttentionWeights {
    pub q: Matrix,
    pub k: Matrix,
    pub v: Matrix,
    pub o: Matrix,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub config: AttentionConfig,
}
#[derive(Debug, Clone)]
pub struct AttentionActivations {
    pub projected_q_gate: Vec<f32>,
    pub projected_key: Vec<f32>,
    pub query_before_rope: Vec<f32>,
    pub key_before_rope: Vec<f32>,
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub gate: Vec<f32>,
    pub scores: Vec<f32>,
    pub probabilities: Vec<f32>,
    pub context: Vec<f32>,
    pub gated: Vec<f32>,
    pub output: Vec<f32>,
}
pub fn rotate(x: &mut [f32], position: usize, dim: usize, theta: f32) -> Result<()> {
    require(
        dim <= x.len() && dim % 2 == 0 && theta.is_finite() && theta > 0.,
        "invalid RoPE",
    )?;
    // Qwenのrotate_halfは隣接ペアではなく、回転部分の前半と後半を組にする。
    // 部分RoPEなので x[dim..] は位置に依存せず、そのまま残す。
    // テキスト専用ではMRoPEのT/H/W位置が等しく、通常の1次元回転に一致する。
    for i in 0..dim / 2 {
        let angle = position as f32 / theta.powf((2 * i) as f32 / dim as f32);
        let (s, c) = angle.sin_cos();
        let (a, b) = (x[i], x[i + dim / 2]);
        x[i] = a * c - b * s;
        x[i + dim / 2] = b * c + a * s;
    }
    finite(x)
}
/// Dense causal softmax/V kernel. Keys have already been RoPE transformed.
pub fn attend(
    query: &[f32],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    c: &AttentionConfig,
) -> Result<Vec<f32>> {
    Ok(attend_traced(query, keys, values, c)?.context)
}
pub struct AttentionRead {
    pub context: Vec<f32>,
    pub scores: Vec<f32>,
    pub probabilities: Vec<f32>,
}
pub fn attend_traced(
    query: &[f32],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    c: &AttentionConfig,
) -> Result<AttentionRead> {
    require(
        c.heads > 0 && c.kv_heads > 0 && c.head_dim > 0 && c.heads % c.kv_heads == 0,
        "invalid GQA",
    )?;
    require(
        query.len() == c.heads * c.head_dim && !keys.is_empty() && keys.len() == values.len(),
        "attention shape mismatch",
    )?;
    finite(query)?;
    for (k, v) in keys.iter().zip(values) {
        require(
            k.len() == c.kv_heads * c.head_dim && v.len() == k.len(),
            "KV shape mismatch",
        )?;
        finite(k)?;
        finite(v)?;
    }
    let mut out = vec![0.; query.len()];
    let mut all_scores = Vec::new();
    let mut all_probabilities = Vec::new();
    for h in 0..c.heads {
        // GQA: 例えばQ=8/KV=2ならQの0..3がKV=0、4..7がKV=1。
        // h % kv_heads とすると共有相手が変わり、別の演算になる。
        let kv = h / (c.heads / c.kv_heads);
        let q = &query[h * c.head_dim..(h + 1) * c.head_dim];
        let scores: Vec<_> = keys
            .iter()
            .map(|k| {
                q.iter()
                    .zip(&k[kv * c.head_dim..(kv + 1) * c.head_dim])
                    .map(|(a, b)| a * b)
                    .sum::<f32>()
                    / (c.head_dim as f32).sqrt()
            })
            .collect();
        // softmaxは「どの過去位置をどれだけ読むか」。各ヘッドで独立に
        // 総和1になり、その確率でVを加重平均する。未来のK/Vは渡さない。
        let probabilities = softmax(&scores)?;
        all_scores.extend_from_slice(&scores);
        all_probabilities.extend_from_slice(&probabilities);
        for (p, v) in probabilities.iter().zip(values) {
            for d in 0..c.head_dim {
                out[h * c.head_dim + d] += p * v[kv * c.head_dim + d];
            }
        }
    }
    finite(&out)?;
    Ok(AttentionRead {
        context: out,
        scores: all_scores,
        probabilities: all_probabilities,
    })
}
impl AttentionWeights {
    fn project(&self, x: &[f32], position: usize, b: Backend) -> Result<AttentionActivations> {
        let c = &self.config;
        let mixed = self.q.project(x, b)?;
        let mut key = self.k.project(x, b)?;
        let value = self.v.project(x, b)?;
        require(
            mixed.len() == c.heads * c.head_dim * 2
                && key.len() == c.kv_heads * c.head_dim
                && value.len() == key.len(),
            "attention projection shape",
        )?;
        let mut query = Vec::new();
        let mut gate = Vec::new();
        // q_proj全体をQ半分/gate半分に切ってはいけない。
        // 格納は [head0_Q, head0_gate, head1_Q, head1_gate, ...]。
        for head in mixed.chunks_exact(c.head_dim * 2) {
            query.extend(rms_norm(
                &head[..c.head_dim],
                &self.q_norm,
                c.eps,
                Scale::QwenZeroCentered,
            )?);
            gate.extend_from_slice(&head[c.head_dim..]);
        }
        let projected_key = key.clone();
        let query_before_rope = query.clone();
        for head in query.chunks_exact_mut(c.head_dim) {
            rotate(head, position, c.rotary_dim, c.theta)?;
        }
        for head in key.chunks_exact_mut(c.head_dim) {
            head.copy_from_slice(&rms_norm(
                head,
                &self.k_norm,
                c.eps,
                Scale::QwenZeroCentered,
            )?);
        }
        let key_before_rope = key.clone();
        for head in key.chunks_exact_mut(c.head_dim) {
            rotate(head, position, c.rotary_dim, c.theta)?;
        }
        Ok(AttentionActivations {
            projected_q_gate: mixed,
            projected_key,
            query_before_rope,
            key_before_rope,
            query,
            key,
            value,
            gate,
            scores: vec![],
            probabilities: vec![],
            context: vec![],
            gated: vec![],
            output: vec![],
        })
    }
    fn finish(
        &self,
        a: &mut AttentionActivations,
        keys: &[Vec<f32>],
        values: &[Vec<f32>],
        b: Backend,
    ) -> Result<()> {
        let read = attend_traced(&a.query, keys, values, &self.config)?;
        let mut out = read.context.clone();
        a.context = read.context;
        a.scores = read.scores;
        a.probabilities = read.probabilities;
        for (v, g) in out.iter_mut().zip(&a.gate) {
            *v *= sigmoid(*g);
        }
        a.output = self.o.project(&out, b)?;
        a.gated = out;
        Ok(())
    }
    pub fn step(
        &self,
        x: &[f32],
        position: usize,
        state: &mut KvCache,
        b: Backend,
    ) -> Result<AttentionActivations> {
        require(
            position == state.keys.len() && state.values.len() == position,
            "KV position mismatch",
        )?;
        // キャッシュに保存するのは位置回転後のKと元のV。次のトークンで
        // 過去のKを再回転せず、Qだけを新しい絶対位置で計算する。
        let mut a = self.project(x, position, b)?;
        state.keys.push(a.key.clone());
        state.values.push(a.value.clone());
        self.finish(&mut a, &state.keys, &state.values, b)?;
        Ok(a)
    }
    pub fn sequence(&self, xs: &[Vec<f32>], b: Backend) -> Result<Vec<AttentionActivations>> {
        let mut all: Vec<_> = xs
            .iter()
            .enumerate()
            .map(|(p, x)| self.project(x, p, b))
            .collect::<Result<_>>()?;
        let keys: Vec<_> = all.iter().map(|a| a.key.clone()).collect();
        let values: Vec<_> = all.iter().map(|a| a.value.clone()).collect();
        for (p, a) in all.iter_mut().enumerate() {
            self.finish(a, &keys[..=p], &values[..=p], b)?;
        }
        Ok(all)
    }
}
