use qwen_rust_inference::{
    ffn::FfnWeights,
    ops::{add, multiply, rms_norm, silu, softmax, Scale},
    tensor::Matrix,
};
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let (mut abs, mut rel) = (0.0_f32, 0.0_f32);
    for (&x, &y) in a.iter().zip(b) {
        let e = (x - y).abs();
        abs = abs.max(e);
        rel = rel.max(e / y.abs().max(1e-12));
        assert!(
            x.is_finite() && e <= 2e-6 + 2e-5 * y.abs(),
            "actual={x} expected={y} max_abs={abs} max_rel={rel}"
        );
    }
    eprintln!("max_abs={abs:e} max_rel={rel:e}");
}
fn mat(r: usize, c: usize, v: &[f32]) -> Matrix {
    Matrix::new(r, c, v.to_vec()).unwrap()
}
#[test]
fn hand_arithmetic() {
    assert_eq!(
        mat(2, 2, &[1., 2., 3., 4.]).linear(&[1., 2.]).unwrap(),
        [5., 11.]
    );
    assert_eq!(add(&[1., 2.], &[3., 4.]).unwrap(), [4., 6.]);
    assert_eq!(multiply(&[1., 2.], &[3., 4.]).unwrap(), [3., 8.]);
    assert_eq!(softmax(&[1000., 1000.]).unwrap(), [0.5, 0.5]);
    close(
        &silu(&[-1000., 0., 1., 1000.]).unwrap(),
        &[0., 0., 0.7310586, 1000.],
    );
    close(
        &rms_norm(&[3., 4.], &[1., 1.], 1e-6, Scale::Direct).unwrap(),
        &[0.8485281, 1.1313709],
    );
    close(
        &rms_norm(&[3., 4.], &[0., 0.], 1e-6, Scale::QwenZeroCentered).unwrap(),
        &[0.8485281, 1.1313709],
    );
}
#[test]
fn invalid_inputs_and_overflow() {
    assert!(Matrix::new(2, 2, vec![1.]).is_err());
    assert!(Matrix::new(0, 2, vec![]).is_err());
    assert!(Matrix::new(usize::MAX, 2, vec![]).is_err());
    assert!(mat(1, 2, &[1., 2.]).linear(&[1.]).is_err());
    assert!(add(&[1.], &[]).is_err());
    assert!(multiply(&[f32::MAX], &[2.]).is_err());
    assert!(mat(1, 2, &[f32::MAX, f32::MAX]).linear(&[2., 2.]).is_err());
    for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(Matrix::new(1, 1, vec![x]).is_err());
        assert!(softmax(&[x]).is_err());
        assert!(silu(&[x]).is_err());
        assert!(rms_norm(&[x], &[1.], 1e-6, Scale::Direct).is_err());
    }
    assert!(softmax(&[]).is_err());
    assert!(rms_norm(&[1.], &[1.], 0., Scale::Direct).is_err());
    assert!(rms_norm(&[1.], &[], 1e-6, Scale::Direct).is_err());
    assert!(rms_norm(&[f32::MAX], &[1.], 1e-6, Scale::Direct).is_err());
}
#[test]
fn hand_ffn_intervention() {
    let f = FfnWeights::new(
        mat(2, 2, &[0., 0., 1., 0.]),
        mat(2, 2, &[1., 0., 0., 1.]),
        mat(2, 2, &[1., 3., 2., 4.]),
    )
    .unwrap();
    let base = f.forward(&[1., 2.], None).unwrap();
    let edited = f.forward(&[1., 2.], Some(1)).unwrap();
    let s = 1.0 / (1.0 + (-1.0_f32).exp());
    close(&base.output, &[6. * s, 8. * s]);
    close(&edited.output, &[0., 0.]);
    close(&base.product, &[0., 2. * s]);
    let col = f.writing_vector(1).unwrap();
    close(
        &base
            .output
            .iter()
            .zip(&edited.output)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>(),
        &col.iter().map(|v| v * base.product[1]).collect::<Vec<_>>(),
    );
    assert_eq!(base, f.forward(&[1., 2.], None).unwrap());
    assert!(f.forward(&[1., 2.], Some(2)).is_err());
    assert!(f.writing_vector(2).is_err());
    assert!(FfnWeights::new(
        mat(1, 2, &[1., 2.]),
        mat(2, 2, &[1.; 4]),
        mat(2, 1, &[1.; 2])
    )
    .is_err());
}
