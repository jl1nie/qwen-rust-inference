use qwen_rust_inference::weights::{decode_tensor, TensorData, TensorReader, TextConfig};
use safetensors::Dtype;
use std::collections::BTreeMap;
#[test]
fn bf16_is_bit_extension_not_integer_conversion() {
    assert_eq!(
        decode_tensor(Dtype::BF16, &[0x80, 0x3f, 0x00, 0xc0, 0, 0]).unwrap(),
        [1., -2., 0.]
    );
    assert!(decode_tensor(Dtype::BF16, &[0x80, 0x7f]).is_err());
    assert!(decode_tensor(Dtype::BF16, &[1]).is_err());
    assert!(decode_tensor(Dtype::I8, &[1]).is_err());
    assert_eq!(
        decode_tensor(Dtype::F32, &1.5_f32.to_le_bytes()).unwrap(),
        [1.5]
    );
}
#[test]
fn missing_wrong_shape_and_unexpected_unused_are_errors() {
    let t = TensorData {
        shape: vec![2, 1],
        values: vec![1., 2.],
    };
    let make = || TensorReader {
        tensors: BTreeMap::from([("weight".into(), t.clone())]),
    };
    assert!(make().take("missing", &[2, 1]).is_err());
    assert!(make().take("weight", &[1, 2]).is_err());
    assert!(make().finish(&[]).is_err());
    assert_eq!(make().finish(&["weight".into()]).unwrap(), ["weight"]);
    assert_eq!(make().take("weight", &[2, 1]).unwrap(), [1., 2.]);
}
#[test]
fn reject_unsupported_config_before_allocating() {
    let v: serde_json::Value = serde_json::from_str(include_str!("fixtures/hybrid.json")).unwrap();
    let mut c: TextConfig = serde_json::from_value(v["config"].clone()).unwrap();
    c.validate().unwrap();
    c.num_key_value_heads = 0;
    assert!(c.validate().is_err());
    c.num_key_value_heads = 1;
    c.rope_parameters.rope_type = "dynamic".into();
    assert!(c.validate().is_err());
    c.rope_parameters.rope_type = "default".into();
    c.tie_word_embeddings = false;
    assert!(c.validate().is_err());
    let config = serde_json::from_value(v["config"].clone()).unwrap();
    let reader = TensorReader {
        tensors: serde_json::from_value(v["tensors"].clone()).unwrap(),
    };
    assert!(qwen_rust_inference::weights::model_from_tensors(
        config,
        reader,
        "test".into(),
        &["model.language_model.norm.weight".into()]
    )
    .unwrap_err()
    .to_string()
    .contains("cannot ignore language"));
}
