
sample_filepath = "data/spin_dists_per_time/N4/a0p763/IC1/L256/N4_a0p763_IC1_L256_z2p0_sample149.csv"
df = CSV.read(sample_filepath, DataFrame)

sample_lambdas = df[!, "lambda"]

