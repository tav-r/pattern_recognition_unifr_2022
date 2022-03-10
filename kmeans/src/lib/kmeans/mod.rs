use std::iter::Sum;
use nalgebra::DVector;

pub mod csv;
pub mod init;
pub mod cluster;

fn distance(v: &DVector<f64>, w: &DVector<f64>) -> f64 {
    Sum::sum(((v - w).transpose() * (v - w)).iter())
}