use crate::{
    ops::{l2_norm, multiply, rms_norm, sigmoid, silu, softplus, Scale},
    state::DeltaState,
    tensor::{finite, require, Backend, Matrix, Result},
};
#[derive(Debug, Clone)]
pub struct DeltaConfig {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub kernel: usize,
    pub eps: f32,
}
#[derive(Debug, Clone)]
pub struct DeltaWeights {
    pub qkv: Matrix,
    pub z: Matrix,
    pub a: Matrix,
    pub b: Matrix,
    pub conv: Matrix,
    pub dt: Vec<f32>,
    pub a_log: Vec<f32>,
    pub norm: Vec<f32>,
    pub out: Matrix,
    pub config: DeltaConfig,
}
#[derive(Debug, Clone)]
pub struct DeltaActivations {
    pub convolution_sums: Vec<f32>,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub z: Vec<f32>,
    pub convolution_state: Vec<f32>,
    pub recurrent_state: Vec<f32>,
    pub projected: Vec<f32>,
    pub convolved: Vec<f32>,
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub decay: Vec<f32>,
    pub beta: Vec<f32>,
    pub core: Vec<f32>,
    pub gated: Vec<f32>,
    pub output: Vec<f32>,
}
/// One head, state row-major [key_dim,value_dim]. Mutates only recurrent state.
pub fn delta_rule(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: f32,
    beta: f32,
    state: &mut [f32],
) -> Result<Vec<f32>> {
    finite(q)?;
    finite(k)?;
    finite(v)?;
    finite(state)?;
    require(
        q.len() == k.len()
            && state.len() == k.len() * v.len()
            && g.is_finite()
            && g <= 0.
            && beta.is_finite()
            && (0.0..=1.0).contains(&beta),
        "invalid delta rule input",
    )?;
    // Sは [key_dim,value_dim] の「連想記憶」。AttentionのKV列とは異なり
    // トークン数によって要素数が増えない。g<=0なのでexp(g)は0..1。
    let decay = g.exp();
    for s in state.iter_mut() {
        *s *= decay;
    }
    // まず古い記憶から k に対応する値 k^T S を読む。
    // 書くのはvそのものではなく、予測との差分 delta=beta*(v-k^T S)。
    // beta=0なら更新なし、beta=1ならこのキーの予測誤差を全量補正する。
    let mut delta = v.to_vec();
    for j in 0..v.len() {
        let memory = (0..k.len())
            .map(|i| state[i * v.len() + j] * k[i])
            .sum::<f32>();
        delta[j] = (delta[j] - memory) * beta;
    }
    // 外積 k delta^T はrank-1更新。添字 i はキー方向、j は値方向。
    // 状態の向きを逆にすると、正方行列の例では気付きにくいので注意。
    for i in 0..k.len() {
        for j in 0..v.len() {
            state[i * v.len() + j] += k[i] * delta[j];
        }
    }
    // 更新後の状態をqで読む。q/sqrt(key_dim)は呼び出し側で計算済み。
    // 同じトークンの書込みも、この位置の出力から参照できる。
    let out = (0..v.len())
        .map(|j| (0..q.len()).map(|i| state[i * v.len() + j] * q[i]).sum())
        .collect::<Vec<f32>>();
    finite(state)?;
    finite(&out)?;
    Ok(out)
}
impl DeltaWeights {
    pub fn new_state(&self) -> DeltaState {
        let c = &self.config;
        DeltaState {
            convolution: vec![0.; self.qkv.rows() * (c.kernel - 1)],
            recurrent: vec![0.; c.value_heads * c.key_dim * c.value_dim],
        }
    }
    pub fn step(
        &self,
        x: &[f32],
        state: &mut DeltaState,
        backend: Backend,
    ) -> Result<DeltaActivations> {
        self.step_captured(x, state, backend, true)
    }
    pub(crate) fn step_captured(
        &self,
        x: &[f32],
        state: &mut DeltaState,
        backend: Backend,
        capture: bool,
    ) -> Result<DeltaActivations> {
        let c = &self.config;
        let channels = self.qkv.rows();
        require(
            c.kernel > 0
                && c.key_heads > 0
                && c.value_heads % c.key_heads == 0
                && c.key_dim > 0
                && c.value_dim > 0,
            "invalid DeltaNet config",
        )?;
        require(
            state.convolution.len() == channels * (c.kernel - 1)
                && state.recurrent.len() == c.value_heads * c.key_dim * c.value_dim,
            "DeltaNet state shape mismatch",
        )?;
        let projected = self.qkv.project(x, backend)?;
        let mut convolved = vec![0.; channels];
        // causal depthwise convolution: チャンネル間は混ぜず時間だけ混ぜる。
        // 履歴は古い順でkernel-1個。最後の係数が現在の射影値に掛かる。
        for ch in 0..channels {
            let weights = self.conv.row(ch)?;
            let history = &mut state.convolution[ch * (c.kernel - 1)..(ch + 1) * (c.kernel - 1)];
            let mut sum = 0.;
            for (v, w) in history.iter().zip(weights.iter()) {
                sum += v * w;
            }
            sum += projected[ch] * weights[c.kernel - 1];
            convolved[ch] = sum;
            if !history.is_empty() {
                history.rotate_left(1);
                let last = history.len() - 1;
                history[last] = projected[ch];
            }
        }
        let convolution_sums = convolved.clone();
        convolved = silu(&convolved)?;
        // 射影の並びは [全Qヘッド, 全Kヘッド, 全Vヘッド]。
        // RMSNormと異なり、Q/Kは「平方平均」でなく「平方和」でL2正規化。
        let kd = c.key_heads * c.key_dim;
        let vd = c.value_heads * c.value_dim;
        require(convolved.len() == 2 * kd + vd, "DeltaNet QKV shape")?;
        let mut query = Vec::new();
        let mut key = Vec::new();
        let value = convolved[2 * kd..].to_vec();
        for h in 0..c.value_heads {
            let kh = h / (c.value_heads / c.key_heads);
            query.extend(l2_norm(&convolved[kh * c.key_dim..(kh + 1) * c.key_dim])?);
            key.extend(l2_norm(
                &convolved[kd + kh * c.key_dim..kd + (kh + 1) * c.key_dim],
            )?);
        }
        for q in &mut query {
            *q /= (c.key_dim as f32).sqrt();
        }
        let a = self.a.project(x, backend)?;
        let b = self.b.project(x, backend)?;
        let z = self.z.project(x, backend)?;
        require(
            a.len() == c.value_heads
                && b.len() == a.len()
                && self.dt.len() == a.len()
                && self.a_log.len() == a.len(),
            "DeltaNet gate shape",
        )?;
        // betaは更新量、gは対数減衰。A_logを直接減衰率として使わず、
        // exp(A_log)*softplus(a+dt_bias)で正の速度を作ってから負号を付ける。
        let beta: Vec<_> = b.iter().map(|v| sigmoid(*v)).collect();
        let decay: Vec<_> = a
            .iter()
            .enumerate()
            .map(|(h, v)| -self.a_log[h].exp() * softplus(v + self.dt[h]))
            .collect();
        let mut core = Vec::new();
        for h in 0..c.value_heads {
            let size = c.key_dim * c.value_dim;
            core.extend(delta_rule(
                &query[h * c.key_dim..(h + 1) * c.key_dim],
                &key[h * c.key_dim..(h + 1) * c.key_dim],
                &value[h * c.value_dim..(h + 1) * c.value_dim],
                decay[h],
                beta[h],
                &mut state.recurrent[h * size..(h + 1) * size],
            )?);
        }
        let mut gated = Vec::new();
        require(z.len() == core.len(), "DeltaNet output gate shape")?;
        // 出力は「直接係数のRMSNorm → SiLU(z)ゲート」。ブロックの
        // zero-centered RMSNormやAttentionのsigmoidゲートとは別の式。
        for (head, gate) in core
            .chunks_exact(c.value_dim)
            .zip(z.chunks_exact(c.value_dim))
        {
            gated.extend(multiply(
                &rms_norm(head, &self.norm, c.eps, Scale::Direct)?,
                &silu(gate)?,
            )?);
        }
        let output = self.out.project(&gated, backend)?;
        Ok(DeltaActivations {
            convolution_sums,
            a,
            b,
            z,
            convolution_state: if capture {
                state.convolution.clone()
            } else {
                vec![]
            },
            recurrent_state: if capture {
                state.recurrent.clone()
            } else {
                vec![]
            },
            projected,
            convolved,
            query,
            key,
            value,
            decay,
            beta,
            core,
            gated,
            output,
        })
    }
}
