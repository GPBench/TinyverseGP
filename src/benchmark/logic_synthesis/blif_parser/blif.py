class BlifFile:
    """BLIF file parser"""

    def __init__(self, known_gates=None):
        self.known_gates = []
        for i in range(len(known_gates)):
            g = known_gates[i]
            if g is None:
                g = "-null-"
            self.known_gates.append(g)
        self.initialize_()

    def parse(self, blif_or_file: str) -> tuple:
        """
        Parses a BLIF file and returns the number of inputs, outputs and gates.
        The parsed data is stored in the object and can be accessed using the eachGate, eachInput and eachOutput methods.

        :param blif_or_file: A string containing the BLIF data or a file object
        :return: A tuple containing the number of inputs, outputs and gates
        """

        if isinstance(blif_or_file, str):
            data = blif_or_file
            assert ".model" in data, "The BLIF file must contain a model."
        else:
            data = blif_or_file.read()

        self.initialize_()

        lbuf = ""
        # NOTE: T.S. Disabled unused code
        # lineid = 0
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.endswith("\\"):
                lbuf += line[:-1]
                continue
            if lbuf:
                line = lbuf + line
                lbuf = ""
            line = line.split("#", 1)[0]
            if not line:
                continue
            if line.startswith("."):
                # param
                wordlist = line[1:].split()
                pid, args = wordlist[0], wordlist[1:]

                if self.tableSpec is not None:
                    self._identifyGate()

                if pid == "names":
                    self.tableSpec = args
                    if len(args) > 3:
                        raise Exception(
                            'Invalid number of inputs (1, 2 or 3 input gates are supported only) "%s"'
                            % line
                        )
                elif pid == "inputs":
                    self.inputs = args
                elif pid == "outputs":
                    self.outputs = args
                elif pid == "model":
                    pass
                elif pid == "end":
                    break
                elif pid.startswith("default"):
                    pass
                elif pid == "gate":
                    args[0] = args[0].upper()
                    if args[0] in self.name_conversion:
                        args[0] = self.name_conversion[args[0]]
                    if args[0] in self.known_gates:
                        gtype = self.known_gates.index(args[0])
                    elif (args[0] == "ZERO") and ("XOR2" in self.known_gates):
                        # constant zero using XOR
                        gtype = self.known_gates.index("XOR2")
                    elif (args[0] == "ONE") and ("XNOR2" in self.known_gates):
                        # constant 1 using XNOR
                        gtype = self.known_gates.index("XNOR2")
                    else:
                        raise Exception(
                            'Unknown gate "%s". Supported gate names: %s'
                            % (args[0], ",".join(self.known_gates))
                        )

                    pin_map = {}
                    for a in args[1:]:
                        pin_name, map = [b.strip() for b in a.split("=")]
                        pin_map[pin_name.upper()] = map

                    if len(pin_map) > 3:
                        raise Exception(
                            'Invalid number of inputs (1,2 or 3 input gates are supported only) "%s"'
                            % line
                        )

                    if "B" not in pin_map:
                        pin_map["B"] = None
                    if "A" not in pin_map:
                        pin_map["A"] = None

                    self.gatelist[pin_map["O"]] = BlifGate(
                        pin_map["O"], pin_map["A"], pin_map["B"], gtype
                    )

                else:
                    raise Exception('Unknown command ".%s"' % pid)

            elif self.tableSpec is not None:
                line = line.split()
                if len(line) == 1:
                    if line[0] == "0":
                        self.tableFunc = 0
                    else:
                        self.tableFunc = 1
                    self.tableData.append((line[0]))
                else:

                    if (line[1] != "1") and (line[1] != "0"):
                        raise Exception(
                            "Invalid BLIF data: %s. Right side should contain 0 for OFF-set or 1 for ON-set"
                            % " ".join(line)
                        )

                    # First data line, initialize tableFunc
                    if self.tableFunc is None:
                        if line[1] == "0":  # OFF-set
                            self.tableFunc = (1 << 2 ** (len(self.tableSpec) - 1)) - 1
                        else:  # ON-set
                            self.tableFunc = 0

                    vals = expand(line[0])
                    for v in vals:
                        self.tableData.append((v, line[1]))

                    expval = 0
                    for v in vals:
                        # print v, int(v,2)
                        expval |= 1 << int(v, 2)

                    if line[1] == "0":  # OFF-set
                        expval = expval ^ ((1 << 2 ** (len(self.tableSpec) - 1)) - 1)
                        self.tableFunc &= expval
                    else:  # ON-set
                        self.tableFunc |= expval

            else:
                raise Exception("Invalid BLIF data: %s" % line)

        if self.tableSpec is not None:
            self._identifyGate()

        # Topological sort of gatelist
        # NOTE: T.S. Disabled unused code
        # remaining = {key: idx for idx, key in enumerate(self.gatelist.keys())}
        g2lev = {}
        for i in self.inputs:
            g2lev[i] = 0

        def prop(g):
            if g in g2lev:
                return g2lev[g]
            ina = self.gatelist[g].ina
            ina = prop(ina) if (ina is not None) else 0
            inb = self.gatelist[g].inb
            inb = prop(inb) if (inb is not None) else 0
            g2lev[g] = max(ina, inb) + 1
            return g2lev[g]

        for o in self.outputs:
            prop(o)

        g2lev = sorted(g2lev.items(), key=lambda x: x[1])
        self.toposorted = [g for g, lev in g2lev if g not in self.inputs]

        return len(self.inputs), len(self.outputs), len(self.gatelist)

    def eachGate(self):
        """Returns an iterator over the gates"""
        for g in self.toposorted:
            yield self.gatelist[g]

    def eachInput(self):
        """Returns an iterator over the inputs"""
        for i in self.inputs:
            yield i

    def eachOutput(self):
        """Returns an iterator over the outputs"""
        for o in self.outputs:
            yield o

    def initialize_(self):
        self.outputs = []
        self.inputs = []
        self.gatelist = {}
        self.toposorted = []
        self.tableSpec = None
        self.tableData = []
        self.tableFunc = None
        self.result = None
        self.name_conversion = {
            "INV1": "INVA",
            "INV": "INVA",
            "NOR": "NOR2",
            "OR": "OR2",
            "AND": "AND2",
            "NAND": "NAND2",
            "XNOR": "XNOR2",
            "XOR": "XOR2",
            "XORA": "XOR2",
            "XORA2": "XOR2",
            "XORB2": "XOR2",
            "XNORA": "XNOR2",
            "XNORA2": "XNOR2",
            "XNORB2": "XNOR2",
            "BUF": "IDA",
            "BUF1": "IDA",
            "WIRE": "IDA",
            "ZERO0": "ZERO",
            "ONE0": "ONE",
        }

    def _identifyGate(self):
        if self.tableSpec is None:
            if len(self.tableData):
                raise Exception("Internal error")
            return

        if len(self.tableSpec) == 1:
            if self.tableFunc is None:
                self.tableFunc = 0

            # const 0 = XOR2, const 1 = XNOR2
            func2gate = {0: "XOR2", 1: "XNOR2"}
            try:
                gtype = self.known_gates.index(func2gate[self.tableFunc])
            except Exception:
                raise Exception(
                    "Unknown constant. Truth table: %s Index: %d"
                    % (str(self.tableData), self.tableFunc)
                )

            self.gatelist[self.tableSpec[0]] = BlifGate(
                self.tableSpec[0], None, None, gtype
            )

        elif len(self.tableSpec) == 2:
            # 1-input, 1-output
            func2gate = {1: "INVA", 2: "IDA"}
            try:
                gtype = self.known_gates.index(func2gate[self.tableFunc])
            except Exception:
                raise Exception(
                    "Unknown 1-input gate. Truth table: %s Index: %d"
                    % (
                        ",".join(["%s:%s" % (a, b) for a, b in self.tableData]),
                        self.tableFunc,
                    )
                )

            self.gatelist[self.tableSpec[1]] = BlifGate(
                self.tableSpec[1], self.tableSpec[0], None, gtype
            )

        else:
            # 2-inputs, 1-output
            func2gate = {
                1: "NOR2",
                6: "XOR2",
                7: "NAND2",
                8: "AND2",
                14: "OR2",
                9: "XNOR2",
                2: "AND2NOTA",
                4: "AND2NOTB",
                11: "OR2NOTA",
                13: "OR2NOTB",
            }
            try:
                gtype = self.known_gates.index(func2gate[self.tableFunc])
            except Exception:
                raise Exception(
                    "Unknown 2-input gate. Truth table: {%s} Function index: %d"
                    % (
                        ",".join(["%s:%s" % (a, b) for a, b in self.tableData]),
                        self.tableFunc,
                    )
                )

            self.gatelist[self.tableSpec[2]] = BlifGate(
                self.tableSpec[2], self.tableSpec[0], self.tableSpec[1], gtype
            )

        self.tableSpec = None
        self.tableData = []
        self.tableFunc = None


class BlifGate:
    def __init__(self, name, ina, inb, fun):
        self.name = name
        self.ina = ina
        self.inb = inb
        self.fun = fun
        self.lev = None
        self.levidx = None

    def arity(self):
        return 0 + (self.ina is not None) + (self.inb is not None)

    def __repr__(self):
        return "Gate<%s>(%s,%s)" % (self.fun, self.ina, self.inb)


def expand(inp):
    i = inp.find("-")
    res = []
    if i > -1:
        s1 = inp[0:i]
        s2 = inp[i + 1 :]
        res.extend(expand(s1 + "0" + s2))
        res.extend(expand(s1 + "1" + s2))
    else:
        res.append(inp)

    return res
