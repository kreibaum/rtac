use std::{fs::File, io, mem};

use dfdx::tensor::HasArrayData;
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

use crate::nn::MultiLayerPerceptron;

/// Import from MessagePack and export to MessagePack

#[derive(Debug, Deserialize, Serialize)]
struct IntermediaryModel {
    #[serde(with = "BigArray")]
    l1w: [f32; 9 * 128],
    #[serde(with = "BigArray")]
    l1b: [f32; 128],
    #[serde(with = "BigArray")]
    l2w: [f32; 128 * 128],
    #[serde(with = "BigArray")]
    l2b: [f32; 128],
    #[serde(with = "BigArray")]
    l3aw: [f32; 128 * 9],
    #[serde(with = "BigArray")]
    l3ab: [f32; 9],
    #[serde(with = "BigArray")]
    l3bw: [f32; 128 * 1],
    #[serde(with = "BigArray")]
    l3bb: [f32; 1],
}

pub fn save_model(mlp: MultiLayerPerceptron, filename: &str) -> Result<(), io::Error> {
    let (l1, _, l2, _, l3) = mlp;
    let ((l3a, _), (l3b, _)) = l3.0;

    let im = unsafe {
        IntermediaryModel {
            l1w: mem::transmute(*l1.weight.data()),
            l1b: *l1.bias.data(),
            l2w: mem::transmute(*l2.weight.data()),
            l2b: *l2.bias.data(),
            l3aw: mem::transmute(*l3a.weight.data()),
            l3ab: *l3a.bias.data(),
            l3bw: mem::transmute(*l3b.weight.data()),
            l3bb: *l3b.bias.data(),
        }
    };

    // Serialize the IntermediaryModel to a file.
    let mut file = File::create(filename)?;

    rmp_serde::encode::write_named(&mut file, &im).unwrap();

    Ok(())
}

pub fn load_model(filename: &str) -> Result<MultiLayerPerceptron, io::Error> {
    // Deserialize the IntermediaryModel from a file.
    let file = File::open(filename)?;
    let im: IntermediaryModel = rmp_serde::decode::from_read(file).unwrap();

    let mut mlp: MultiLayerPerceptron = Default::default();

    // Un-flatten all the Tensors from one dimensional vectors.
    let l1w: [[f32; 9]; 128] = unsafe { mem::transmute(im.l1w) };
    let l1b: [f32; 128] = im.l1b;

    let l2w: [[f32; 128]; 128] = unsafe { mem::transmute(im.l2w) };
    let l2b: [f32; 128] = im.l2b;

    let l3aw: [[f32; 128]; 9] = unsafe { mem::transmute(im.l3aw) };
    let l3ab: [f32; 9] = im.l3ab;

    let l3bw: [[f32; 128]; 1] = unsafe { mem::transmute(im.l3bw) };
    let l3bb: [f32; 1] = im.l3bb;

    // Load the Tensors into the MultiLayerPerceptron.
    mlp.0.weight.mut_data().copy_from_slice(&l1w);
    mlp.0.bias.mut_data().copy_from_slice(&l1b);

    mlp.2.weight.mut_data().copy_from_slice(&l2w);
    mlp.2.bias.mut_data().copy_from_slice(&l2b);

    mlp.4 .0 .0 .0.weight.mut_data().copy_from_slice(&l3aw);
    mlp.4 .0 .0 .0.bias.mut_data().copy_from_slice(&l3ab);

    mlp.4 .0 .1 .0.weight.mut_data().copy_from_slice(&l3bw);
    mlp.4 .0 .1 .0.bias.mut_data().copy_from_slice(&l3bb);

    Ok(mlp)
}
