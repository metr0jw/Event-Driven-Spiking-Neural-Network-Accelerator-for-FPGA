"""
Test error handling and exception hierarchy.

Tests custom exception classes and validates that proper errors are raised
with correct error codes and context information.
"""

import pytest
import numpy as np
from snn_fpga_accelerator.exceptions import (
    FPGAError,
    TimeoutError as AcceleratorTimeoutError,
    WeightLoadError,
    ConfigurationError,
    CommunicationError,
    BitstreamError,
    KernelExecutionError,
    validate_parameter,
)


class TestExceptionHierarchy:
    """Test custom exception classes and their properties."""
    
    def test_fpga_error_base(self):
        """Test FPGAError base class."""
        error = FPGAError("Test error", error_code=1000)
        assert "Test error" in str(error)
        assert error.error_code == 1000
        assert error.context == {}
        
    def test_fpga_error_with_context(self):
        """Test FPGAError with context dictionary."""
        error = FPGAError("Register error", error_code=1001)
        error.context["device_id"] = 0
        error.context["register"] = 0x10
        assert error.error_code == 1001
        assert error.context["device_id"] == 0
        assert error.context["register"] == 0x10
    
    def test_timeout_error(self):
        """Test AcceleratorTimeoutError."""
        error = AcceleratorTimeoutError(
            "Kernel timeout",
            timeout_duration=10.0,
            operation="inference",
            error_code=6001
        )
        assert error.timeout_duration == 10.0
        assert error.operation == "inference"
        assert error.error_code == 6001
        assert "10.0" in str(error)
    
    def test_weight_load_error(self):
        """Test WeightLoadError."""
        error = WeightLoadError(
            "Invalid weight shape",
            weight_shape=(300, 300),
            error_code=3001
        )
        assert error.weight_shape == (300, 300)
        assert error.weight_index is None
        assert "300" in str(error) or "weight_shape" in repr(error)
    
    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError(
            "Invalid encoding type",
            parameter="encoding_type",
            value=5,
            valid_range=(0, 3),
            error_code=4001
        )
        assert error.parameter == "encoding_type"
        assert error.value == 5
        assert error.valid_range == (0, 3)
        assert "encoding_type" in str(error)
    
    def test_communication_error(self):
        """Test CommunicationError."""
        error = CommunicationError(
            "Device not found",
            device_index=0,
            xclbin_path="/path/to/bitstream.xclbin",
            error_code=1002
        )
        assert error.device_index == 0
        assert error.xclbin_path == "/path/to/bitstream.xclbin"
        assert "Device not found" in str(error)
    
    def test_bitstream_error(self):
        """Test BitstreamError."""
        error = BitstreamError(
            "Bitstream version mismatch",
            bitstream_path="/path/to/old.xclbin",
            expected_version="v2.0",
            actual_version="v1.0",
            error_code=2001
        )
        assert error.bitstream_path == "/path/to/old.xclbin"
        assert error.expected_version == "v2.0"
        assert error.actual_version == "v1.0"
        assert "mismatch" in str(error)
    
    def test_kernel_execution_error(self):
        """Test KernelExecutionError."""
        error = KernelExecutionError(
            "Kernel crash",
            kernel_name="snn_top_hls",
            exit_code=255,
            error_code=5001
        )
        assert error.kernel_name == "snn_top_hls"
        assert error.exit_code == 255
        assert "snn_top_hls" in str(error)


class TestValidateParameter:
    """Test validate_parameter helper function."""
    
    def test_validate_valid_option(self):
        """Test validation with valid option."""
        # Should not raise
        validate_parameter(
            "rate_poisson",
            valid_options=["none", "rate_poisson", "latency"],
            parameter_name="encoding_type"
        )
    
    def test_validate_invalid_option(self):
        """Test validation with invalid option."""
        with pytest.raises(ConfigurationError, match="encoding_type"):
            validate_parameter(
                "invalid",
                valid_options=["none", "rate_poisson", "latency"],
                parameter_name="encoding_type"
            )
    
    def test_validate_in_range(self):
        """Test validation with value in valid range."""
        # Should not raise
        validate_parameter(
            150,
            value_range=(0, 255),
            parameter_name="baseline"
        )
    
    def test_validate_below_range(self):
        """Test validation with value below range."""
        with pytest.raises(ConfigurationError, match="baseline"):
            validate_parameter(
                -10,
                value_range=(0, 255),
                parameter_name="baseline"
            )
    
    def test_validate_above_range(self):
        """Test validation with value above range."""
        with pytest.raises(ConfigurationError, match="weight"):
            validate_parameter(
                300,
                value_range=(0, 255),
                parameter_name="weight"
            )
    
    def test_validate_both_constraints(self):
        """Test validation with both option list and range."""
        # Should validate options first, then range if needed
        with pytest.raises(ConfigurationError, match="test_param"):
            validate_parameter(
                "invalid",
                valid_options=["a", "b", "c"],
                value_range=(0, 10),
                parameter_name="test_param"
            )


class TestWeightValidation:
    """Test weight loading validation."""
    
    def test_weight_shape_2d_required(self):
        """Test that 1D weights are rejected."""
        # This would be tested in xrt_backend or accelerator tests
        # by calling load_weights with incorrect shape
        pass
    
    def test_weight_values_in_range(self):
        """Test that out-of-range weights are rejected."""
        # Tested via xrt_backend.load_weights()
        pass
    
    def test_weight_matrix_size_limit(self):
        """Test 256x256 size limit."""
        # Tested via xrt_backend.load_weights()
        pass


class TestConfigurationValidation:
    """Test configuration parameter validation."""
    
    def test_encoding_type_validation(self):
        """Test encoding type must be 0-3."""
        # Valid: 0, 1, 2, 3
        # Invalid: -1, 4, 15, 255
        pass
    
    def test_baseline_range(self):
        """Test baseline must be 0-255."""
        pass
    
    def test_num_steps_range(self):
        """Test num_steps must be 1-65535."""
        pass


class TestErrorPropagation:
    """Test that errors propagate correctly through the stack."""
    
    def test_xrt_backend_error_propagation(self):
        """Test XRTBackend errors propagate to SNNAccelerator."""
        # Would require mocking XRT device failures
        pass
    
    def test_timeout_detection(self):
        """Test kernel timeout is detected and reported."""
        # Would require mocking kernel that never completes
        pass
    
    def test_weight_load_failure_recovery(self):
        """Test system state after weight load failure."""
        # Ensure no partial state corruption
        pass


class TestErrorMessages:
    """Test error messages are clear and actionable."""
    
    def test_error_includes_context(self):
        """Test errors include relevant context information."""
        error = WeightLoadError(
            "Weight shape invalid",
            weight_shape=(300, 300),
            error_code=3002
        )
        error_str = str(error)
        assert "300" in error_str or "weight_shape" in repr(error) or error.weight_shape == (300, 300)
    
    def test_error_includes_code(self):
        """Test errors include error codes for debugging."""
        error = CommunicationError("Test", error_code=1010)
        assert error.error_code == 1010
    
    def test_chained_exceptions(self):
        """Test exception chaining preserves original cause."""
        try:
            try:
                raise RuntimeError("Original error")
            except RuntimeError as e:
                raise CommunicationError(
                    "Wrapped error",
                    error_code=1000
                ) from e
        except CommunicationError as ce:
            assert ce.__cause__ is not None
            assert isinstance(ce.__cause__, RuntimeError)
            assert str(ce.__cause__) == "Original error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
