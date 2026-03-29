# from crane.preprocess.electrode_subset import subset_electrodes


# def test_batch_subset(make_batch):
#     batch = make_batch(n_elec=5)
#     out = subset_electrodes(batch, max_n_electrodes=2)
#     # correct shape and channels
#     assert out["ieeg"]["data"].shape[1] == 2


# def test_batch_inplace(make_batch):
#     batch = make_batch(n_elec=5)
#     out = subset_electrodes(batch, max_n_electrodes=2, inplace=False)
#     # correct shape and channels
#     assert out["ieeg"]["data"].shape[1] == 2
#     # original batch not modified
#     assert batch["ieeg"]["data"].shape[1] == 5
#     assert batch is not out

#     subset_electrodes(batch, max_n_electrodes=2, inplace=True)
#     # original batch modified
#     assert batch["ieeg"]["data"].shape[1] == 2


# def test_noop_subset(make_batch):
#     batch = make_batch(n_elec=3)  # less than max_n_electrodes
#     out = subset_electrodes(batch, max_n_electrodes=5)
#     # correct shape and channels
#     assert out["ieeg"]["data"].shape[1] == 3
