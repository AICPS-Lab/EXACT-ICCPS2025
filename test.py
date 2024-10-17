from datasets.PhysiQ import PhysiQ


if __name__ == "__main__":
    # test the dataset

    dataset = PhysiQ(root="data", N_way=2, split="train")
    print(len(dataset))
    print(dataset[0])
