def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    pixel_mask   = [item["pixel_mask"] for item in batch]
    labels       = [item["labels"] for item in batch]

    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "labels": labels
    }
