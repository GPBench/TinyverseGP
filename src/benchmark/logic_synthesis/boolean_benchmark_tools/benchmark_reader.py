"""
Provides a reader that can be used as an interface for the benchmarks of the
General Boolean Function Benchmark Suite (GBFS):
https://dl.acm.org/doi/abs/10.1145/3594805.3607131

The architecture is based on three classes implemented in the file
- TruthTable: Dataclass to represent a compressed or uncompressed truth table
- Benchmark: Represent a logic synthesis benchmark with a truth table and meta information
- BenchmarkReader: Reads and stores the actual data of a given benchmark file
"""

from os import path
from dataclasses import dataclass
import array


@dataclass
class TruthTable:
    """
    Class for representing a compressed or uncompressed truth table.
    Input and outputs are stores row-wise in two-dimensional vectors.
    """

    inputs: list
    outputs: list
    input_names: list
    output_names: list
    compressed: bool

    def __init__(self):
        self.inputs = []
        self.outputs = []

        self.input_names = []
        self.output_names = []

        self.compressed = False

    def clear(self):
        """
        Clears the vectors of the input and output vectors.

        :return: None
        """
        self.inputs.clear()
        self.outputs.clear()
        self.input_names.clear()
        self.output_names.clear()


class Benchmark:
    """
    Class for representing a logic synthesis benchmark.
    """

    def __init__(self):
        self.table = TruthTable()

        # Variables for the meta information of a certain benchmark
        self.num_inputs = -1
        self.num_outputs = -1
        self.num_chunks = -1
        self.num_product_terms = -1

        self.model_name = ""
        self.header_size = -1

    def clear(self) -> None:
        """
        Clears the truth table

        :return: None
        """
        self.table.clear()

    def rows(self) -> int:
        """
        Return the number of rows of the table.

        :return: Number of rows.
        """
        return len(self.table.inputs)

    def append_inputs(self, input_row: list) -> None:
        """
        Appends a new input row vector to the inputs.
        Validates the input row vector by using assert.
        The input row vector must be not None and non-empty.

        :param input_row: Vector containing a row of inputs
        :return: None
        """
        assert input_row is not None and len(input_row) > 0, (
            "Input row vector is" "None or empty!"
        )
        self.table.inputs.append(input_row)

    def append_outputs(self, output_row: list) -> None:
        """
        Appends a new output row vector to the outputs.
        Validates the output row vector by using assert.
        The output row vector must be not None and non-empty.

        :param output_row: Vector containing a row of outputs
        :return: None
        """
        assert output_row is not None and len(output_row) > 0, (
            "Output row vector is " "None or emtpy!"
        )
        self.table.outputs.append(output_row)

    def get_inputs_at(self, index: int) -> list:
        """
        Returns a row of inputs at a specific index.
        Validates the index by using assert. The index
        must be in the interval 0 <= index <= max_index.

        :param index: index of the row
        :return: Vector containing a row of inputs
        """
        max_index = len(self.table.inputs)
        assert 0 <= index <= max_index, "Index is out of range!"
        return self.table.inputs[index]

    def get_outputs_at(self, index: int) -> list:
        """
        Returns a row of outputs at a specific index.
        Validates the index by using assert. The index
        must be in the interval 0 <= index <= max_index.

        :param index: index of the row
        :return: Vector containing a row of outputs
        """
        max_index = len(self.table.outputs)
        assert 0 <= index <= max_index, "Index is out of range!"
        return self.table.outputs[index]

    def print_input_names(self) -> None:
        print("Input names: ", end="")
        for name in self.table.input_names:
            print(name + " ", end="")

        print("")

    def print_output_names(self) -> None:
        print("Output names: ", end="")
        for name in self.table.output_names:
            print(name + " ", end="")

        print("")

    def print_header(self) -> None:
        if len(self.model_name) > 0:
            print("Model: %s " % self.model_name)

        if self.num_inputs != -1:
            print("Inputs: %d" % self.num_inputs)

        if self.num_outputs != -1:
            print("Outputs: %d" % self.num_outputs)

        if len(self.table.input_names) > 0:
            self.print_input_names()

        if len(self.table.output_names) > 0:
            self.print_output_names()

    def print(self) -> None:
        """
        Print the table row-wise.

        :return: None
        """
        # Get the number of rows
        num_rows = self.rows()

        # Iterate over the number of rows
        for i in range(num_rows):

            # Print the inputs that are stored in the current
            # input row vector
            for input_row in self.table.inputs[i]:
                print(input_row + " ", end="")

            # Separate inputs and outs with whitespace
            print("   ", end="")

            # Print the outputs of the current row then
            for output_row in self.table.outputs[i]:
                print(output_row + " ", end="")

            print("")


class BenchmarkReader:
    """
    The BenchmarkReader class provides methods for validating and reading
    PLU and TT benchmark files.

    The inputs and outputs are stored in an instance of the TruthTable class.
    """

    def __init__(self):
        # Constants for the status of the file format
        self.file = None
        self.PLU = 0
        self.TT = 1

        # Create an TruthTable instance for the storage of the table data
        self.benchmark = Benchmark()

    def validate_file_path(self, file_path: str) -> None:
        """
        Validates the file which is adressed by the given path by checking
        whether the path is empty, the file exists, is a file and can be read.

        The file extension has to match those of plu and pla files (i.e. .plu .pla .PLU .PLA)

        :param file_path: Given path of the file which will be validated
        :return: None
        """

        # Check the length of the file path, raise Exception when empty
        if len(file_path) == 0:
            raise Exception("File path is empty!")

        # Validate whether the file exists
        if not path.exists(file_path):
            raise Exception("File does not exists!")

        # Check if the path addresses a file
        if not path.isfile(file_path):
            raise Exception("File path does not address a file!")

        # Get the file extension
        filename, extension = path.splitext(file_path)

        # Validate the extension by checking if it's a PLU file
        if extension.lower() not in [".plu", ".tt"]:
            raise Exception("File type is not valid for benchmark reader!")

    def file_format(self, file_path: str) -> int:
        """
        Checks the file extension and returns the file format (PLU or PLA).

        :param file_path: Path of the file of which the extension will be checked
        :return: value of the respective file format constant
        """

        # Get the file extension
        filename, extension = path.splitext(file_path)

        # Check whether the extension is plu or pla
        if extension.lower() == ".plu":
            return self.PLU
        elif extension.lower() == ".tt":
            return self.TT
        else:
            # Raise Exception is the file is not a PLU or PLA file
            raise Exception("Given file is not valid for benchmark reader!")

    def open_file(self, file_path: str) -> None:
        self.validate_file_path(file_path)
        self.file = open(file_path, "r")

    def close_file(
        self,
    ) -> None:
        self.file.close()

    def read_keyword(self, keyword: str) -> str:
        if len(keyword) == 0:
            raise Exception("Keyword is empty!")

        self.file.seek(0)
        line = ""

        for line in self.file:
            if keyword in line:
                break

        # Just perform the default split the current line by whitespace
        line_values = line.split()

        if len(line_values) > 2:
            raise Exception("Not a valid keyword line")

        return line_values[1]

    def read_num_inputs(self) -> None:
        self.benchmark.num_inputs = int(self.read_keyword(".i"))

    def read_num_outputs(self) -> None:
        self.benchmark.num_outputs = int(self.read_keyword(".o"))

    def read_num_product_terms(self) -> None:
        self.benchmark.num_product_terms = int(self.read_keyword(".p"))

    def read_model_name(self) -> None:
        self.benchmark.model_name = self.read_keyword(".model")

    def read_names(self, keyword: str) -> array:
        if len(keyword) == 0:
            raise Exception("Keyword is empty!")

        self.file.seek(0)

        line_values = []

        for line in self.file:
            if keyword in line:
                line_values = line.split()
                break

        if len(line_values) <= 1:
            raise Exception("Not a valid keyword line")

        return line_values

    def read_input_names(self) -> None:
        line_values = self.read_names(".ilb")
        self.benchmark.table.input_names = list(line_values)
        self.benchmark.table.input_names.pop()

    def read_output_names(self) -> None:
        line_values = self.read_names(".ob")
        self.benchmark.table.output_names = list(line_values)
        self.benchmark.table.output_names.pop()

    def read_header(self) -> None:
        header_size = 0

        self.read_num_inputs()

        if self.benchmark.num_inputs != -1:
            header_size += 1

        self.read_num_outputs()

        if self.benchmark.num_outputs != -1:
            header_size += 1

        self.read_model_name()

        if len(self.benchmark.model_name) > 0:
            header_size += 1

        self.read_input_names()

        if len(self.benchmark.table.input_names) > 0:
            header_size += 1

        self.read_output_names()

        if len(self.benchmark.table.output_names) > 0:
            header_size += 1

        self.benchmark.header_size = header_size

    def read_tt_file(self, file_path: str) -> None:
        self.benchmark.table.compressed = False

        # First, validate the file path
        self.validate_file_path(file_path)

        input_row = []
        output_row = []
        line = ""

        # Open the file for reading
        self.open_file(file_path)
        self.read_header()

        i = 0
        while i <= self.benchmark.header_size:
            line = self.file.readline()
            i += 1

        # Read in the input/output data until the end of the
        # table data is reached
        while ".end" not in line:
            # Just perform the default split the current line by whitespace
            line_split = line.split()

            inputs = line_split[0]
            outputs = line_split[1]

            for index in range(len(inputs)):
                input_row.append(inputs[index])

            for index in range(len(outputs)):
                output_row.append(outputs[index])

            # Store the data of the row in the vectors of the truth table
            self.benchmark.append_inputs(input_row.copy())
            self.benchmark.append_outputs(output_row.copy())

            # Remove the input/output row data from the temporary storage
            input_row.clear()
            output_row.clear()

            # Continue the iteration with the next line of the file
            line = self.file.readline()

    def read_plu_file(self, file_path: str) -> None:
        """
        Reads a PLU benchmark file. First, the header is read and
        then the data of the compressed/uncompressed truth table is read line by line.

        The method automatically distinguishes between a PLU and PLA file.

        The procedures for the reading process differ in the header size and the format
        of each row of the truth table.

        :param file_path: Path of the PLU/TT file
        :return: None
        """

        # First, validate the file path
        self.validate_file_path(file_path)

        # Lists for the header and input/output rows
        header = []
        input_row = []
        output_row = []

        # Set header size depending on the status of the file format
        header_size = 3
        self.benchmark.compressed = True

        # Open the file for reading
        self.open_file(file_path)

        # Read in the header
        i = 0
        while i < header_size:
            line = self.file.readline()
            identifier, value = line.split()
            header.append(value)
            i += 1

        # Store the meta information of the header
        # in the respective member variables
        self.benchmark.num_inputs = int(header[0])
        self.benchmark.num_outputs = int(header[1])
        self.benchmark.num_chunks = int(header[2])

        # Read in the input/output data until the end of the
        # table data is reached
        line = self.file.readline()
        while ".e" not in line:

            # Just perform the default split the current line by whitespace
            line_values = line.split()

            # Iterate over the values of the current line
            for index, value in enumerate(line_values):
                # Depending on the number of inputs and outputs,
                # store them in the respective row vectors
                if index < int(self.benchmark.num_inputs):
                    input_row.append(value)
                else:
                    output_row.append(value)

            # Store the data of the row in the vectors of the truth table
            self.benchmark.append_inputs(input_row.copy())
            self.benchmark.append_outputs(output_row.copy())

            # Remove the input/output row data from the temporary storage
            input_row.clear()
            output_row.clear()

            # Continue the iteration with the next line of the file
            line = self.file.readline()
        self.close_file()
