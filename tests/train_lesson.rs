//! The FFN backward lesson (volume 1, chapter 4's example) against its
//! closed forms.
#![cfg(feature = "train")]
use qwen_rust_inference::train::lesson::ffn_backward;

fn close(a: &[f32], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((*x as f64 - y).abs() < 1e-6, "{a:?} vs {b:?}");
    }
}

#[test]
fn ffn_backward_lesson_has_its_closed_forms() {
    let l = ffn_backward(1e-3).unwrap();
    let s = 1. / (1. + (-1_f64).exp()); // σ(1)
    let silu1 = s; // SiLU(1) = 1·σ(1)
    let d1 = s * (1. + (1. - s)); // SiLU'(1)
    assert!((l.s - silu1).abs() < 1e-12 && (l.silu_prime_1 - d1).abs() < 1e-12);
    close(&l.g, &[0., 1.]);
    close(&l.u, &[1., 2.]);
    close(&l.a, &[0., 2. * silu1]);
    close(&l.y, &[6. * silu1, 8. * silu1]);
    // Only the component volume 1, chapter 8 removes by hand: row 2, column 2.
    close(&l.d_down, &[0., 0., 0., 2. * silu1]);
    close(&l.da, &[2., 4.]);
    close(&l.dg, &[2. * 1. * 0.5, 4. * 2. * d1]);
    close(&l.du, &[0., 4. * silu1]);
    close(&l.d_gate, &[1., 2., 8. * d1, 16. * d1]);
    close(&l.d_up, &[0., 0., 4. * silu1, 8. * silu1]);
    // W_gateᵀ dg + du: (2·1 − 1·8d1 + 0, −1·1 + 1·8d1 + 4s)
    close(&l.dx, &[2. - 8. * d1, -1. + 8. * d1 + 4. * silu1]);
    assert!(l.numerical_max_error < 1e-5, "{}", l.numerical_max_error);
    // A small step lowers y2 by about lr·‖∇‖².
    let drop = (l.y2_before - l.y2_after) as f64;
    let predicted = l.y2_before as f64 - l.y2_predicted;
    assert!(
        drop > 0. && (drop - predicted).abs() < 0.05 * predicted,
        "{drop} vs {predicted}"
    );
    assert!(l.text().contains("only row 2, column 2"));
}
