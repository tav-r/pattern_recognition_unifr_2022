mod lib;

use lib::kmeans::{csv,cluster};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>>{
    let train = csv::load_csv("train.csv")?;
    let clustering = cluster::deterministic_kmeans(
        train.rows(1, train.nrows()-1)
        .column_iter()
        .map(|r| r.into()).collect(), 15
    );

    for (i, cluster) in clustering.iter().enumerate() {
        println!("Cluster {} has {} members", i, cluster.len())
    }

    Ok(())
}