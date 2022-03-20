mod lib;

use lib::kmeans::{csv,cluster,indices};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>>{
    let train = csv::load_csv("train.csv")?;
    let train_vecs = train.rows(1, train.nrows()-1)
        .column_iter()
        .map(|r| r.into()).collect();

    for k in [5, 7, 9, 10, 12, 15] {
        println!("k={}", k);

        let clustering = cluster::deterministic_kmeans(
            &train_vecs, k
        );

        println!("Davis-Bouldin-Index of clustering {}", indices::davis_bouldin_index(&clustering));
        println!("Dunn-Index of clustering {}", indices::dunn_index(&clustering));

        println!();
    }

    Ok(())
}