"""
Custom Exception Classes for SNN FPGA Accelerator.

This module defines custom exceptions for better error handling and debugging:
- FPGAError: Base exception for all FPGA-related errors
- TimeoutError: Operations that exceed time limits
- WeightLoadError: Weight loading/validation failures
- ConfigurationError: Invalid configuration parameters
- CommunicationError: XRT/hardware communication failures
"""


class FPGAError(Exception):
    """Base exception class for all FPGA accelerator errors.
    
    All custom exceptions inherit from this class to allow catching
    any accelerator-related error with a single except clause.
    
    Attributes:
        message: Human-readable error description
        error_code: Optional numeric error code
        context: Optional dict with additional error context
    """
    
    def __init__(self, message: str, error_code: int = None, context: dict = None):
        """Initialize FPGAError.
        
        Args:
            message: Error description
            error_code: Optional numeric error code
            context: Optional dict with additional context (e.g., register values)
        """
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        
        # Build detailed error message
        full_message = message
        if error_code is not None:
            full_message = f"[Error {error_code}] {message}"
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            full_message = f"{full_message} (Context: {context_str})"
        
        super().__init__(full_message)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(message={self.message!r}, error_code={self.error_code}, context={self.context})"


class TimeoutError(FPGAError):
    """Exception raised when an operation exceeds its time limit.
    
    Examples:
        - Kernel execution timeout
        - DMA transfer timeout
        - Register read/write timeout
    
    Attributes:
        timeout_duration: Duration in seconds before timeout
        operation: Name of the operation that timed out
    """
    
    def __init__(self, message: str, timeout_duration: float = None, operation: str = None, **kwargs):
        """Initialize TimeoutError.
        
        Args:
            message: Error description
            timeout_duration: Timeout duration in seconds
            operation: Operation name (e.g., "kernel_run", "dma_transfer")
            **kwargs: Additional context passed to FPGAError
        """
        self.timeout_duration = timeout_duration
        self.operation = operation
        
        context = kwargs.get('context', {})
        if timeout_duration is not None:
            context['timeout_duration'] = f"{timeout_duration}s"
        if operation is not None:
            context['operation'] = operation
        
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class WeightLoadError(FPGAError):
    """Exception raised when weight loading or validation fails.
    
    Examples:
        - Invalid weight dimensions
        - Weight values out of range
        - DMA transfer failure during weight upload
        - Checksum mismatch after loading
    
    Attributes:
        weight_shape: Shape of weights that failed to load
        weight_index: Index of problematic weight (if applicable)
    """
    
    def __init__(self, message: str, weight_shape: tuple = None, weight_index: int = None, **kwargs):
        """Initialize WeightLoadError.
        
        Args:
            message: Error description
            weight_shape: Shape tuple (e.g., (784, 128))
            weight_index: Index of problematic weight
            **kwargs: Additional context passed to FPGAError
        """
        self.weight_shape = weight_shape
        self.weight_index = weight_index
        
        context = kwargs.get('context', {})
        if weight_shape is not None:
            context['weight_shape'] = str(weight_shape)
        if weight_index is not None:
            context['weight_index'] = weight_index
        
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class ConfigurationError(FPGAError):
    """Exception raised for invalid configuration parameters.
    
    Examples:
        - Invalid encoding type
        - Out-of-range parameter values
        - Incompatible parameter combinations
        - Missing required configuration
    
    Attributes:
        parameter: Name of invalid parameter
        value: Invalid value provided
        valid_range: Valid range or options
    """
    
    def __init__(self, message: str, parameter: str = None, value=None, valid_range=None, **kwargs):
        """Initialize ConfigurationError.
        
        Args:
            message: Error description
            parameter: Name of invalid parameter
            value: Invalid value that was provided
            valid_range: Valid range/options (str, list, or tuple)
            **kwargs: Additional context passed to FPGAError
        """
        self.parameter = parameter
        self.value = value
        self.valid_range = valid_range
        
        context = kwargs.get('context', {})
        if parameter is not None:
            context['parameter'] = parameter
        if value is not None:
            context['value'] = str(value)
        if valid_range is not None:
            context['valid_range'] = str(valid_range)
        
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class CommunicationError(FPGAError):
    """Exception raised for XRT/hardware communication failures.
    
    Examples:
        - XRT device not found
        - XCLBIN loading failure
        - Register read/write failure
        - DMA buffer allocation failure
        - AXI bus error
    
    Attributes:
        device_index: FPGA device index (if applicable)
        xclbin_path: Path to XCLBIN file (if applicable)
    """
    
    def __init__(self, message: str, device_index: int = None, xclbin_path: str = None, **kwargs):
        """Initialize CommunicationError.
        
        Args:
            message: Error description
            device_index: FPGA device index
            xclbin_path: Path to XCLBIN file
            **kwargs: Additional context passed to FPGAError
        """
        self.device_index = device_index
        self.xclbin_path = xclbin_path
        
        context = kwargs.get('context', {})
        if device_index is not None:
            context['device_index'] = device_index
        if xclbin_path is not None:
            context['xclbin_path'] = xclbin_path
        
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class BitstreamError(FPGAError):
    """Exception raised for bitstream-related errors.
    
    Examples:
        - Bitstream file not found
        - Invalid bitstream format
        - Bitstream loading failure
        - Version mismatch
    
    Attributes:
        bitstream_path: Path to bitstream file
        expected_version: Expected bitstream version
        actual_version: Actual bitstream version
    """
    
    def __init__(self, message: str, bitstream_path: str = None, 
                 expected_version: str = None, actual_version: str = None, **kwargs):
        """Initialize BitstreamError.
        
        Args:
            message: Error description
            bitstream_path: Path to bitstream file
            expected_version: Expected version string
            actual_version: Actual version string
            **kwargs: Additional context passed to FPGAError
        """
        self.bitstream_path = bitstream_path
        self.expected_version = expected_version
        self.actual_version = actual_version
        
        context = kwargs.get('context', {})
        if bitstream_path is not None:
            context['bitstream_path'] = bitstream_path
        if expected_version is not None:
            context['expected_version'] = expected_version
        if actual_version is not None:
            context['actual_version'] = actual_version
        
        kwargs['context'] = context
        super().__init__(message, **kwargs)


class KernelExecutionError(FPGAError):
    """Exception raised when kernel execution fails.
    
    Examples:
        - Kernel launch failure
        - Kernel hang/deadlock
        - Kernel internal error
        - Invalid kernel arguments
    
    Attributes:
        kernel_name: Name of kernel that failed
        exit_code: Kernel exit code (if available)
    """
    
    def __init__(self, message: str, kernel_name: str = None, exit_code: int = None, **kwargs):
        """Initialize KernelExecutionError.
        
        Args:
            message: Error description
            kernel_name: Name of kernel (e.g., "snn_top_hls")
            exit_code: Kernel exit code
            **kwargs: Additional context passed to FPGAError
        """
        self.kernel_name = kernel_name
        self.exit_code = exit_code
        
        context = kwargs.get('context', {})
        if kernel_name is not None:
            context['kernel_name'] = kernel_name
        if exit_code is not None:
            context['exit_code'] = exit_code
        
        kwargs['context'] = context
        super().__init__(message, **kwargs)


# Convenience function to validate parameters
def validate_parameter(value, valid_options=None, value_range=None, parameter_name="parameter"):
    """Validate a parameter value and raise ConfigurationError if invalid.
    
    Args:
        value: Value to validate
        valid_options: List/set of valid options (for discrete values)
        value_range: Tuple (min, max) for numeric range validation
        parameter_name: Name of parameter for error message
    
    Raises:
        ConfigurationError: If validation fails
    
    Examples:
        >>> validate_parameter('rate', ['rate', 'latency', 'delta_sigma'], parameter_name='encoding_type')
        >>> validate_parameter(100, value_range=(1, 1000), parameter_name='num_steps')
    """
    if valid_options is not None:
        if value not in valid_options:
            raise ConfigurationError(
                f"Invalid {parameter_name}: {value!r}",
                parameter=parameter_name,
                value=value,
                valid_range=list(valid_options) if not isinstance(valid_options, list) else valid_options
            )
    
    if value_range is not None:
        min_val, max_val = value_range
        if not (min_val <= value <= max_val):
            raise ConfigurationError(
                f"{parameter_name} out of range: {value} not in [{min_val}, {max_val}]",
                parameter=parameter_name,
                value=value,
                valid_range=f"[{min_val}, {max_val}]"
            )
