import grain.python as grain
import pickle
import numpy as np


class ParseRecord(grain.MapTransform):
    def map(self, record: bytes):  # pyrefly:ignore
        data = pickle.loads(record)
        return {"latent": data["latent"], "label": data["label"]}


def create_dataloader(file_path, num_workers, batch_size, seed):
    source = grain.ArrayRecordDataSource(file_path)

    index_sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=None,
        shard_options=grain.ShardOptions(
            shard_index=0, shard_count=1, drop_remainder=True
        ),
        shuffle=True,
        seed=seed,
    )

    operations = [
        ParseRecord(),
        grain.Batch(batch_size=batch_size, drop_remainder=True),
    ]

    dataloader = grain.DataLoader(
        data_source=source,
        sampler=index_sampler,
        worker_count=num_workers,
        operations=operations,
    )

    return dataloader
