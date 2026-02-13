from datalair import Lair, download_supplementary_from_geo, Dataset
from pathlib import Path
import anndata as ad
import pandas as pd
from scipy.io import mmread
import tarfile
import tempfile


class DatasetscRNASeqValidation(Dataset):
    """Datalair Dataset class for validation Datasets in this package."""

    def __init__(self) -> None:
        """Initialize this dataset class as a datalair.Dataset class with namespace `DatasetscRNASeqSignature`."""
        super().__init__(namespace="DatasetscRNASeqValidation")


class HippenPerformanceComputationalAlgorithms2023(DatasetscRNASeqValidation):

    def derive(self, lair: Lair) -> None:
        output_dir = lair.get_path(self)
        download_supplementary_from_geo("GSE217517", output_dir)


def read_mtx_tsv_single_cell_data(dir: Path, matrix_name: str) -> ad.AnnData:
    metadata_name = matrix_name.replace("mtx", "tsv")
    obs = pd.read_csv(dir.joinpath(metadata_name.replace("matrix", "barcodes")), sep="\t", header=None)
    obs.columns = obs.columns.astype(str)
    obs.index = obs.index.astype(str)
    var = (
        pd.read_csv(dir.joinpath(metadata_name.replace("matrix", "features")), sep="\t", header=None)
        .rename(columns={0: "ensembl_id", 1: "gene_name", 2: "gene_type"})
    )
    var.columns = var.columns.astype(str)
    var.index = var.index.astype(str)

    return ad.AnnData(
        X = mmread(dir.joinpath(matrix_name)).tocsr().T,
        obs = obs,
        var = var
    )


class HippenPerformanceComputationalAlgorithms2023Adata(DatasetscRNASeqValidation):

    def derive(self, lair: Lair) -> None:
        output_dir = lair.get_path(self)
        ds = HippenPerformanceComputationalAlgorithms2023()
        lair.safe_derive(ds)
        filepaths = lair.get_dataset_filepaths(ds)

        with tempfile.TemporaryDirectory() as tmpdir:
            extract_dir = Path(tmpdir)
            with tarfile.open(filepaths["GSE217517_RAW.tar"], 'r:*') as tar:  # 'r:*' auto-detects compression
                tar.extractall(path=extract_dir)
            filepaths = sorted([path.name for path in extract_dir.iterdir()])
            bulk_filenames = list(filter(lambda x: "bulk" in x, filepaths))
            adatas = []
            for bulk_filename in bulk_filenames:
                df = (
                    pd.read_csv(extract_dir.joinpath(bulk_filename), sep="\t", index_col=0, header=None)
                    .drop(["N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"]).T
                )
                df.columns = df.columns.astype(str)
                df.index = df.index.astype(str)
                adata = ad.AnnData(df)
                adata.var.columns = adata.var.columns.astype(str)
                adata.var.index = adata.var.index.astype(str)
                adata.var.index.name = "index"
                adata.obs.columns = adata.obs.columns.astype(str)
                adata.obs.index = adata.obs.index.astype(str)
                adata.obs.index.name = "index"
                adata.obs["sample_id"] = bulk_filename.split("_")[-2]
                adata.obs["sample_id"] = adata.obs["sample_id"].astype("category")
                adata.obs["geo_id"] = bulk_filename.split("_")[0]
                adata.obs["geo_id"] = adata.obs["geo_id"].astype("category")
                adata.obs["dissociated"] = True if "dissociated" in bulk_filename else False
                adata.obs["depletion"] = "ribo" if "ribo" in bulk_filename else "polyA"
                adata.obs["dissociated"] = adata.obs["dissociated"].astype("category")
                adatas.append(adata)
            var = adata.var.copy()
            adata = ad.concat(adatas, join="outer")
            adata.var = var
            adata.var.columns = adata.var.columns.astype(str)
            adata.var.index = adata.var.index.astype(str)
            adata.var.index.name = "ensembl_id"
            adata.obs.reset_index(inplace=True, drop=True)
            adata.write(output_dir.joinpath("bulk.h5ad"))

            single_cell_filenames = list(filter(lambda x: "single_cell_matrix" in x, filepaths))
            adatas = []
            for single_cell_filename in single_cell_filenames:
                adata = read_mtx_tsv_single_cell_data(extract_dir, single_cell_filename)
                adata.var.columns = adata.var.columns.astype(str)
                adata.var.index = adata.var.index.astype(str)
                adata.var.index.name = "index"
                adata.obs.columns = adata.obs.columns.astype(str)
                adata.obs.index = adata.obs.index.astype(str)
                adata.obs.index.name = "index"
                adata.obs["sample_id"] = single_cell_filename.split("_")[-1].removesuffix(".mtx.gz")
                adata.obs["sample_id"] = adata.obs["sample_id"].astype("category")
                adata.obs["geo_id"] = single_cell_filename.split("_")[0]
                adata.obs["geo_id"] = adata.obs["geo_id"].astype("category")
                adata.obs["pooled"] = True if "pooled" in single_cell_filename else False
                adatas.append(adata)
            var = adata.var.copy()
            adata = ad.concat(adatas, join="outer")
            adata.var = var
            adata.var.columns = adata.var.columns.astype(str)
            adata.var.index = adata.var.index.astype(str)
            adata.var.index.name = "index"
            adata.obs.columns = adata.obs.columns.astype(str)
            adata.obs.index = adata.obs.index.astype(str)
            adata.obs.index.name = "index"
            adata.obs.reset_index(inplace=True, drop=True)
            adata.obs.rename(columns={"0": "cell_id"}, inplace=True)
            adata.write(output_dir.joinpath("single_cell.h5ad"))