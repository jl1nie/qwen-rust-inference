#![cfg(feature = "train")]
use qwen_rust_inference::train::probe::{fit, held_out_groups, ProbeConfig};

/// A tiny deterministic generator (no rand dependency in tests).
fn noise(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
}

/// `classes` centers in `dim` dimensions; each group adds its own offset
/// (another wording) and noise.
fn data(
    classes: usize,
    groups: usize,
    dim: usize,
    signal: f32,
) -> (Vec<f32>, Vec<usize>, Vec<usize>) {
    let mut s = 7;
    let centers: Vec<Vec<f32>> = (0..classes)
        .map(|_| (0..dim).map(|_| noise(&mut s)).collect())
        .collect();
    let (mut x, mut y, mut g) = (Vec::new(), Vec::new(), Vec::new());
    for gi in 0..groups {
        let offset: Vec<f32> = (0..dim).map(|_| noise(&mut s)).collect();
        for (c, center) in centers.iter().enumerate() {
            for k in 0..dim {
                x.push(signal * center[k] + offset[k] + 0.3 * noise(&mut s));
            }
            y.push(c);
            g.push(gi);
        }
    }
    (x, y, g)
}

#[test]
fn a_probe_reads_a_label_the_vectors_encode_in_every_wording() {
    let (x, y, g) = data(6, 4, 16, 3.0);
    let per = held_out_groups(&x, 16, &y, &g, 6, ProbeConfig::default()).unwrap();
    let (c, n) = per.iter().fold((0, 0), |(a, b), &(c, n)| (a + c, b + n));
    assert_eq!(n, 24);
    assert!(c >= 22, "{c}/{n}");
}

#[test]
fn a_probe_cannot_read_a_label_the_vectors_do_not_encode() {
    // No signal: the training wordings are still separable, the held-out one is not.
    let (x, y, g) = data(6, 4, 16, 0.0);
    let train: Vec<usize> = (0..y.len()).filter(|&i| g[i] != 0).collect();
    let tx: Vec<f32> = train
        .iter()
        .flat_map(|&i| x[i * 16..(i + 1) * 16].to_vec())
        .collect();
    let ty: Vec<usize> = train.iter().map(|&i| y[i]).collect();
    let p = fit(
        &tx,
        16,
        &ty,
        6,
        ProbeConfig {
            steps: 500,
            lr: 0.05,
            l2: 0.0,
        },
    )
    .unwrap();
    let fitted = train
        .iter()
        .filter(|&&i| p.predict(&x[i * 16..(i + 1) * 16]) == y[i])
        .count();
    assert_eq!(fitted, train.len());
    let per = held_out_groups(&x, 16, &y, &g, 6, ProbeConfig::default()).unwrap();
    let c: usize = per.iter().map(|p| p.0).sum();
    assert!(c <= 10, "{c}/24 with no signal");
}

#[test]
fn a_probe_needs_two_wordings_and_matching_shapes() {
    let (x, y, _) = data(3, 1, 4, 1.0);
    assert!(held_out_groups(&x, 4, &y, &vec![0; y.len()], 3, ProbeConfig::default()).is_err());
    assert!(fit(&x[1..], 4, &y, 3, ProbeConfig::default()).is_err());
    assert!(fit(&x, 4, &y, 2, ProbeConfig::default()).is_err());
}
