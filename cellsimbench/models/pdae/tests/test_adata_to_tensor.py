import pytest
import anndata
import numpy as np
import torch
import json
from pathlib import Path
from pdae.adata_to_tensor import _get_tensor, _get_perturbation_ohe, map_adata_split_to_tensors
from scipy.sparse import csr_matrix


@pytest.fixture
def mock_adata():
    """Create a mock AnnData object for testing."""
    obs = {
        "condition": ["ctrl", "ctrl", "pert1", "pert2", "pert1+pert2", "ctrl", "pert2", "pert1+pert3"],
        "split": ["train", "train", "train", "train_holdout", "validation", "test", "test", "test"],
        "custom_split": ["tr", "tr", "tr", "val", "val", "te", "te", "te"]
    }
    # Create some synthetic gene expression data
    X = np.random.rand(8, 100)
    return anndata.AnnData(X=X, obs=obs)


@pytest.fixture
def mock_save_dir(tmp_path):
    """Create a temporary directory for saving test data."""
    return tmp_path / "test_tensors"


@pytest.fixture
def mock_adata_path(mock_adata, tmp_path):
    """Save mock AnnData to a temporary file and return the path."""
    dir = tmp_path / "adata"
    dir.mkdir(exist_ok=True)
    adata_path = dir / "mock_adata.h5ad"
    mock_adata.write(adata_path)
    return adata_path


class TestHelperFunctions:
    """Test the helper functions used by map_adata_split_to_tensors."""
    
    def test_get_tensor(self, mock_adata):
        """Test _get_tensor function."""
        pert_labels_unique = ["ctrl", "pert1", "pert2"]
        tensor = _get_tensor(mock_adata, pert_labels_unique, "condition")
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.float32
        # Should have as many rows as observations with these perturbations
        expected_rows = sum(mock_adata.obs["condition"].isin(pert_labels_unique))
        assert tensor.shape[0] == expected_rows
        assert tensor.shape[1] == mock_adata.shape[1]
        # Check if ctrl is correctly represented as zeros
        assert (tensor[pert_labels_unique=="ctrl"] == 0).all()  
    
    def test_get_perturbation_ohe_single_perturbations(self):
        """Test _get_perturbation_ohe with single perturbations."""
        pert_labels = ["ctrl", "pert1", "pert2"]
        dict_single_pert_ix = {"pert1": 0, "pert2": 1, "pert3": 2}
        reference_perturbation = "ctrl"
        pert_sep = "+"
        
        ohe_tensor = _get_perturbation_ohe(pert_labels, dict_single_pert_ix, reference_perturbation, pert_sep)
        
        assert isinstance(ohe_tensor, torch.Tensor)
        assert ohe_tensor.shape == (3, 3)
        assert ohe_tensor.dtype == torch.float32
        
        # Check specific encodings
        expected = torch.tensor([
            [0., 0., 0.],  # ctrl -> zero vector
            [1., 0., 0.],  # pert1
            [0., 1., 0.]   # pert2
        ])
        assert torch.allclose(ohe_tensor, expected)
    
    def test_get_perturbation_ohe_combination_perturbations(self):
        """Test _get_perturbation_ohe with combination perturbations."""
        pert_labels = ["ctrl", "pert1+pert2", "pert1+pert3"]
        dict_single_pert_ix = {"pert1": 0, "pert2": 1, "pert3": 2}
        reference_perturbation = "ctrl"
        pert_sep = "+"
        
        ohe_tensor = _get_perturbation_ohe(pert_labels, dict_single_pert_ix, reference_perturbation, pert_sep)
        
        expected = torch.tensor([
            [0., 0., 0.],  # ctrl -> zero vector
            [1., 1., 0.],  # pert1+pert2
            [1., 0., 1.]   # pert1+pert3
        ])
        assert torch.allclose(ohe_tensor, expected)


class TestMapAdataSplitToTensors:
    """Test the main map_adata_split_to_tensors function."""
    
    def test_basic_functionality(self, mock_adata_path, mock_save_dir):
        """Test basic functionality with all splits."""
        result = map_adata_split_to_tensors(
            adata_path=str(mock_adata_path),
            pert_col="condition",
            pert_sep="+",
            pert_ctrl="ctrl",
            split_col="split",
            train_split="train",
            val_split="validation",
            test_split="test",
            data_dir=str(mock_save_dir)
        )
        
        # Check return structure
        assert isinstance(result, dict)
        required_keys = ["observations", "perturbations", "is_reference_perturbation", "splits", "perturbation_annotations"]
        for key in required_keys:
            assert key in result
        
        # Check tensor shapes and types
        assert isinstance(result["observations"], torch.Tensor)
        assert isinstance(result["perturbations"], torch.Tensor)
        assert isinstance(result["is_reference_perturbation"], torch.Tensor)
        assert isinstance(result["splits"], list)
        assert isinstance(result["perturbation_annotations"], list)
        
        # Check that observations and perturbations have same number of rows
        assert result["observations"].shape[0] == result["perturbations"].shape[0]
        assert result["observations"].shape[0] == len(result["splits"])
        assert result["observations"].shape[0] == len(result["perturbation_annotations"])
        
        # Check that reference perturbation detection works
        assert result["is_reference_perturbation"].dtype == torch.bool
        assert result["is_reference_perturbation"].sum() > 0  # Should have some reference perturbations
    
    def test_with_custom_split_names(self, mock_adata_path, mock_save_dir):
        """Test with custom split column and names."""
        result = map_adata_split_to_tensors(
            adata_path=str(mock_adata_path),
            pert_col="condition",
            pert_sep="+",
            pert_ctrl="ctrl",
            split_col="custom_split",
            train_split="tr",
            val_split="val",
            test_split="te",
            data_dir=str(mock_save_dir)
        )
        
        # Check that all split names are present
        splits_set = set(result["splits"])
        assert "tr" in splits_set
        assert "val" in splits_set
        assert "te" in splits_set
    
    def test_minimal_splits(self, mock_adata_path, mock_save_dir):
        """Test with only train split (minimal requirement)."""
        result = map_adata_split_to_tensors(
            adata_path=str(mock_adata_path),
            pert_col="condition",
            pert_sep="+",
            pert_ctrl="ctrl",
            split_col="split",
            train_split="train",
            data_dir=str(mock_save_dir)
        )
        
        # Should still work with just train split
        assert len(set(result["splits"])) == 1
        assert result["splits"][0] == "train"
    
    def test_files_created(self, mock_adata_path, mock_save_dir):
        """Test that required files are created in data_dir."""
        map_adata_split_to_tensors(
            adata_path=str(mock_adata_path),
            pert_col="condition",
            pert_sep="+",
            pert_ctrl="ctrl",
            split_col="split",
            train_split="train",
            val_split="validation",
            test_split="test",
            data_dir=str(mock_save_dir)
        )
        
        # Check that required files are created
        assert (mock_save_dir / "dict_single_pert_ix.json").exists()
        assert (mock_save_dir / "pert_ohe_str_mapping.json").exists()
        
        # Check file contents
        with open(mock_save_dir / "dict_single_pert_ix.json", "r") as f:
            dict_single_pert_ix = json.load(f)
        
        assert isinstance(dict_single_pert_ix, dict)
        assert len(dict_single_pert_ix) > 0
        assert "ctrl" not in dict_single_pert_ix  # Reference perturbation should not be in index
        
        with open(mock_save_dir / "pert_ohe_str_mapping.json", "r") as f:
            pert_ohe_str_mapping = json.load(f)
        
        assert isinstance(pert_ohe_str_mapping, dict)
        assert len(pert_ohe_str_mapping) > 0
    
    def test_perturbation_encoding_consistency(self, mock_adata_path, mock_save_dir):
        """Test that perturbation encodings are consistent."""
        result = map_adata_split_to_tensors(
            adata_path=str(mock_adata_path),
            pert_col="condition",
            pert_sep="+",
            pert_ctrl="ctrl",
            split_col="split",
            train_split="train",
            val_split="validation",
            test_split="test",
            data_dir=str(mock_save_dir)
        )
        
        # Check that reference perturbations are correctly identified
        ctrl_mask = [ann == "ctrl" for ann in result["perturbation_annotations"]]
        ref_mask = result["is_reference_perturbation"].numpy()
        
        # All ctrl annotations should be marked as reference perturbations
        for i, is_ctrl in enumerate(ctrl_mask):
            if is_ctrl:
                assert ref_mask[i], f"ctrl perturbation at index {i} not marked as reference"
        
        # Check that perturbation tensor has correct dimensions
        with open(mock_save_dir / "dict_single_pert_ix.json", "r") as f:
            dict_single_pert_ix = json.load(f)
        
        assert result["perturbations"].shape[1] == len(dict_single_pert_ix)


class TestErrorCases:
    """Test error handling and edge cases."""
    
    def test_missing_train_split_error(self, mock_adata_path, mock_save_dir):
        """Test that missing train_split raises an assertion error."""
        with pytest.raises(AssertionError, match="train_split must be specified"):
            map_adata_split_to_tensors(
                adata_path=str(mock_adata_path),
                pert_col="condition",
                pert_sep="+",
                pert_ctrl="ctrl",
                split_col="split",
                train_split=None,
                data_dir=str(mock_save_dir)
            )
    
    def test_invalid_split_column_error(self, mock_adata_path, mock_save_dir):
        """Test that invalid split column raises an assertion error."""
        with pytest.raises(AssertionError, match="Split column 'nonexistent' not found"):
            map_adata_split_to_tensors(
                adata_path=str(mock_adata_path),
                pert_col="condition",
                pert_sep="+",
                pert_ctrl="ctrl",
                split_col="nonexistent",
                train_split="train",
                data_dir=str(mock_save_dir)
            )
    
    def test_invalid_pert_column_error(self, mock_adata_path, mock_save_dir):
        """Test that invalid perturbation column raises an assertion error."""
        with pytest.raises(AssertionError, match="Perturbation column 'nonexistent' not found"):
            map_adata_split_to_tensors(
                adata_path=str(mock_adata_path),
                pert_col="nonexistent",
                pert_sep="+",
                pert_ctrl="ctrl",
                split_col="split",
                train_split="train",
                data_dir=str(mock_save_dir)
            )
    
    def test_directory_creation(self, mock_adata_path, tmp_path):
        """Test that the function creates the data directory if it doesn't exist."""
        new_dir = tmp_path / "new_directory"
        assert not new_dir.exists()
        
        map_adata_split_to_tensors(
            adata_path=str(mock_adata_path),
            pert_col="condition",
            pert_sep="+",
            pert_ctrl="ctrl",
            split_col="split",
            train_split="train",
            data_dir=str(new_dir)
        )
        
        assert new_dir.exists()
        assert (new_dir / "dict_single_pert_ix.json").exists()


class TestDataConsistency:
    """Test data consistency and correctness."""
    
    def test_tensor_data_consistency(self, mock_adata_path, mock_save_dir):
        """Test that tensor data is consistent with original AnnData."""
        # Load original data
        adata = anndata.read_h5ad(mock_adata_path)
        
        result = map_adata_split_to_tensors(
            adata_path=str(mock_adata_path),
            pert_col="condition",
            pert_sep="+",
            pert_ctrl="ctrl",
            split_col="split",
            train_split="train",
            val_split="validation",
            test_split="test",
            data_dir=str(mock_save_dir)
        )
        
        # Check that the total number of observations matches
        total_obs_in_splits = len([obs for obs in adata.obs["split"] 
                                 if obs in ["train", "validation", "test"]])
        assert result["observations"].shape[0] == total_obs_in_splits
        
        # Check that gene dimensions match
        assert result["observations"].shape[1] == adata.shape[1]
    
    def test_perturbation_annotation_consistency(self, mock_adata_path, mock_save_dir):
        """Test that perturbation annotations are consistent with original data."""
        result = map_adata_split_to_tensors(
            adata_path=str(mock_adata_path),
            pert_col="condition",
            pert_sep="+",
            pert_ctrl="ctrl",
            split_col="split",
            train_split="train",
            val_split="validation",
            test_split="test",
            data_dir=str(mock_save_dir)
        )
        
        # Check that all perturbation annotations are strings
        assert all(isinstance(ann, str) for ann in result["perturbation_annotations"])
        
        # Check that split annotations match the number of observations
        assert len(result["splits"]) == result["observations"].shape[0]
        assert len(result["perturbation_annotations"]) == result["observations"].shape[0]


class TestDirectoryHandling:
    """Test directory creation and file handling."""
    
    def test_creates_directory_if_not_exists(self, mock_adata_path, tmp_path):
        """Test that the function creates the data directory if it doesn't exist."""
        new_dir = tmp_path / "nonexistent_directory"
        assert not new_dir.exists()
        
        # The function should create the directory
        result = map_adata_split_to_tensors(
            adata_path=str(mock_adata_path),
            pert_col="condition",
            pert_sep="+",
            pert_ctrl="ctrl",
            split_col="split",
            train_split="train",
            data_dir=str(new_dir)
        )
        
        assert new_dir.exists()
        assert (new_dir / "dict_single_pert_ix.json").exists()
        assert (new_dir / "pert_ohe_str_mapping.json").exists()
    
    def test_with_pathlib_data_dir(self, mock_adata_path, tmp_path):
        """Test that the function works with pathlib.Path objects for data_dir."""
        save_dir = tmp_path / "pathlib_test"
        save_dir.mkdir()
        
        result = map_adata_split_to_tensors(
            adata_path=str(mock_adata_path),
            pert_col="condition",
            pert_sep="+",
            pert_ctrl="ctrl",
            split_col="split",
            train_split="train",
            data_dir=save_dir  # Pass as Path object
        )
        
        assert (save_dir / "dict_single_pert_ix.json").exists()
        assert (save_dir / "pert_ohe_str_mapping.json").exists()


class TestJsonSerialization:
    """Test JSON file content and serialization."""
    
    def test_dict_single_pert_ix_content(self, mock_adata_path, mock_save_dir):
        """Test the content of dict_single_pert_ix.json."""
        mock_save_dir.mkdir(parents=True, exist_ok=True)
        
        map_adata_split_to_tensors(
            adata_path=str(mock_adata_path),
            pert_col="condition",
            pert_sep="+",
            pert_ctrl="ctrl",
            split_col="split",
            train_split="train",
            val_split="validation",
            test_split="test",
            data_dir=str(mock_save_dir)
        )
        
        with open(mock_save_dir / "dict_single_pert_ix.json", "r") as f:
            dict_single_pert_ix = json.load(f)
        
        # Should contain perturbations from mock data except reference
        expected_perts = {"pert1", "pert2", "pert3"}  # Based on mock_adata fixture
        assert set(dict_single_pert_ix.keys()) == expected_perts
        
        # Values should be unique indices
        indices = list(dict_single_pert_ix.values())
        assert len(indices) == len(set(indices))  # All unique
        assert all(isinstance(idx, int) for idx in indices)
        assert min(indices) == 0
        assert max(indices) == len(indices) - 1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_with_train_holdout_split(self, mock_adata_path, mock_save_dir):
        """Test with train_holdout_split parameter."""
        mock_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create custom mock data with train_holdout split
        adata = anndata.read_h5ad(mock_adata_path)
        
        # Save modified adata
        modified_path = mock_save_dir / "modified_adata.h5ad"
        adata.write(modified_path)
        
        result = map_adata_split_to_tensors(
            adata_path=str(modified_path),
            pert_col="condition",
            pert_sep="+",
            pert_ctrl="ctrl",
            split_col="split",
            train_split="train",
            train_holdout_split="train_holdout",
            val_split="validation",
            data_dir=str(mock_save_dir)
        )
        
        # Check that train_holdout split is included
        splits_set = set(result["splits"])
        assert "train_holdout" in splits_set
    
    def test_with_only_reference_perturbations(self, tmp_path):
        """Test behavior when only reference perturbations are present in a split."""
        # Create mock data with only ctrl perturbations in test split
        obs = {
            "condition": ["ctrl", "pert1", "ctrl"],
            "split": ["train", "train", "test"]
        }
        X = np.random.rand(3, 50)
        adata = anndata.AnnData(X=X, obs=obs)
        
        adata_path = tmp_path / "ref_only_adata.h5ad"
        adata.write(adata_path)
        
        save_dir = tmp_path / "ref_only_test"
        save_dir.mkdir()
        
        result = map_adata_split_to_tensors(
            adata_path=str(adata_path),
            pert_col="condition",
            pert_sep="+",
            pert_ctrl="ctrl",
            split_col="split",
            train_split="train",
            test_split="test",
            data_dir=str(save_dir)
        )
        
        # Should work even with only reference perturbations in test split
        assert result["observations"].shape[0] == 3
        test_mask = [split == "test" for split in result["splits"]]
        test_ref_mask = result["is_reference_perturbation"][test_mask]
        assert test_ref_mask.all()  # All test observations should be reference
    
    def test_empty_perturbation_strings(self, tmp_path):
        """Test handling of edge cases in perturbation parsing."""
        # Create data with some edge case perturbation strings
        obs = {
            "condition": ["ctrl", "pert1", "pert1+pert2", "pert1++pert2"],  # Note the double +
            "split": ["train", "train", "train", "test"]
        }
        X = np.random.rand(4, 50)
        adata = anndata.AnnData(X=X, obs=obs)
        
        adata_path = tmp_path / "edge_case_adata.h5ad"
        adata.write(adata_path)
        
        save_dir = tmp_path / "edge_case_test"
        save_dir.mkdir()
        
        # This should handle the double separator case
        result = map_adata_split_to_tensors(
            adata_path=str(adata_path),
            pert_col="condition",
            pert_sep="+",
            pert_ctrl="ctrl",
            split_col="split",
            train_split="train",
            test_split="test",
            data_dir=str(save_dir)
        )
        
        # Should still work, treating empty strings as no perturbation
        assert result["observations"].shape[0] == 4


class TestDataTypes:
    """Test data type consistency and conversions."""
    
    def test_tensor_dtypes(self, mock_adata_path, mock_save_dir):
        """Test that output tensors have correct data types."""
        mock_save_dir.mkdir(parents=True, exist_ok=True)
        
        result = map_adata_split_to_tensors(
            adata_path=str(mock_adata_path),
            pert_col="condition",
            pert_sep="+",
            pert_ctrl="ctrl",
            split_col="split",
            train_split="train",
            val_split="validation",
            test_split="test",
            data_dir=str(mock_save_dir)
        )
        
        # Check tensor data types
        assert result["observations"].dtype == torch.float32
        assert result["perturbations"].dtype == torch.float32
        assert result["is_reference_perturbation"].dtype == torch.bool
        
        # Check list/string types
        assert all(isinstance(split, str) for split in result["splits"])
        assert all(isinstance(ann, str) for ann in result["perturbation_annotations"])
    
    def test_sparse_matrix_handling(self, tmp_path):
        """Test that sparse matrices are properly converted to dense tensors."""
        
        # Create mock data with sparse matrix
        X_dense = np.random.rand(4, 50)
        X_sparse = csr_matrix(X_dense)
        
        obs = {
            "condition": ["ctrl", "pert1", "pert2", "ctrl"],
            "split": ["train", "train", "test", "test"]
        }
        
        adata = anndata.AnnData(X=X_sparse, obs=obs)
        adata_path = tmp_path / "sparse_adata.h5ad"
        adata.write(adata_path)
        
        save_dir = tmp_path / "sparse_test"
        save_dir.mkdir()
        
        result = map_adata_split_to_tensors(
            adata_path=str(adata_path),
            pert_col="condition",
            pert_sep="+",
            pert_ctrl="ctrl",
            split_col="split",
            train_split="train",
            test_split="test",
            data_dir=str(save_dir)
        )
        
        # Should successfully convert sparse to dense
        assert result["observations"].shape == (4, 50)
        assert not result["observations"].is_sparse
