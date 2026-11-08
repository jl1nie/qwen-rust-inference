use qwen_rust_inference::validation::compare;
#[test]
fn tolerance_is_per_element_and_first_failure_is_localized() {
    // A huge second reference value must not hide error near zero in the first.
    let e = compare(
        &[0., 100_000.],
        &[0.1, 100_000.],
        0.01,
        0.001,
        "token=63 layer=3",
    )
    .unwrap_err();
    assert!(e
        .to_string()
        .contains("first divergence: token=63 layer=3[0]"));
    assert!(compare(&[1., f32::NAN], &[1., 2.], 1., 1., "fixture")
        .unwrap_err()
        .to_string()
        .contains("fixture[1]"));
}
#[test]
fn error_metrics_and_boundary_tolerance_are_preserved() {
    let m = compare(&[1.5, 2.], &[1., 2.], 0.5, 0., "hand").unwrap();
    assert_eq!(m.max_absolute, 0.5);
    assert_eq!(m.max_relative, 0.5);
    assert_eq!(m.elements, 2);
    let expected_cosine = 5.5_f64 / (6.25_f64 * 5.).sqrt();
    assert!((m.cosine - expected_cosine).abs() < 1e-12);
    assert!(compare(&[1.5, 2.], &[1., 2.], 0.49, 0., "hand").is_err());
    assert_eq!(
        compare(&[0., 0.], &[0., 0.], 0., 0., "zero")
            .unwrap()
            .cosine,
        1.
    );
}
