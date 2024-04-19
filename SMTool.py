import json
import numpy as np
import os
from typing import Literal
from collections.abc import Callable, Iterable
import warnings

SHAPEID_DUCK_HOLDER = "b050533f-200b-4253-9532-c2cfa273f982"
SHAPEID_LOGIC_GATE = "9f0f56e8-2c31-4d83-996c-d00a9b296c3f"
SHAPEID_TIMER = "8f7fd0e7-c46e-4944-a414-7ce2437bb30f"
LOGIC_GATE_MODES = { "and": 0, "or": 1, "xor": 2, "nand": 3, "nor": 4, "xnor": 5 }
LOGIC_GATE_MODE_TO_NAME = { LOGIC_GATE_MODES[key]: key for key in LOGIC_GATE_MODES }
SHAPEID_HEADLIGHT = "ed27f5e2-cac5-4a32-a5d9-49f116acc6af"
SHAPEID_SWITCH = "7cf717d7-d167-4f2d-a6e7-6b2c70aa3986"
SHAPEID_BUTTON = "1e8d93a4-506b-470d-9ada-9c0a321e2db5"
SHAPEID_GLASS_BLOCK = "5f41af56-df4c-4837-9b3c-10781335757f"
SHAPEID_THRUSTER = "a736ffdf-22c1-40f2-8e40-988cab7c0559"
SHAPEID_METAL_BLOCK_1 = "8aedf6c2-94e1-4506-89d4-a0227c552f1e"
SHAPEID_BARRIER_BLOCK = "09ca2713-28ee-4119-9622-e85490034758"
SHAPEID_PLASTIC_BLOCK = "628b2d61-5ceb-43e9-8334-a4135566df7a"

DIRECTION_PLUSY_PLUSZ = (3, 1)
"""
positive Y positive Z
"xaxis": 3,
"zaxis": 1
"""

DIRECTION_NEGX_PLUSZ = (-2, -1)
"""
negative X positive Z
"xaxis": -2,
"zaxis": -1
"""

DIRECTION_PLUSX = (2, 1)
"""
"xaxis": 2,
"zaxis": 1
"""

DIRECTION_IDK_LOL = (3, -2)
"""
"xaxis": 3,
"zaxis": -2
"""

DIRECTION_5 = (1, 3)
"""
"xaxis": 1,
"zaxis": 3
"""

class Blueprint(dict):
    """Scrap Mechanic Blueprint Class"""
    def __init__(self, file: str | None = None) -> None:
        r"""(*** i've never used this so idk if its working lmao ***) Use the file argument to load an existing blueprint in a particular path, example:

            car = Blueprint(r"C:\Users\YourUserName\AppData\Roaming\Axolot Games\Scrap Mechanic\User\User_1234\Blueprints\475db678-e788-4234-b649-f33106890e04\blueprint.json")

            (*** this works ***) If you want to create a new blueprint, don't pass any arguments"""
        if file == None:
            super().__init__(self, bodies=[{"childs": []}], version=3)
        else:
            inputFile = open(file, "r")
            inputJson = json.load(inputFile)
            inputFile.close()
            super().__init__(inputJson)

    def load_list_block(self, list):
        super().clear()
        super().__init__(self, bodies=[{"childs": list}], version=3)


    def get_path():
        """WORKS FOR WINDOWS USERS ONLY !
        
        Iterates through %APPDATA% and returns a list of all the directories behind the given path
        
        Then get's the [1] item of that list, which should be the blueprint folder."""
        return [x[0] for x in os.walk(os.getenv('APPDATA')+"\\Axolot Games\\Scrap Mechanic\\User")][1]


    
    def blueprint_search(self,search_term):
        """WORKS FOR WINDOWS USERS ONLY !

        Uses a search term to retrieve the blueprint file of a creation ! WITH A SPECIFIC NAME.lower() !"""
        blueprints = [x[0] for x in os.walk(self.get_path()+"\\Blueprints")]
        for i in range(len(blueprints)):
            if i==0:
                continue
            try:
                with open(blueprints[i]+"\\description.json") as f:
                    data = json.load(f)
            except:
                #raise TypeError("This blueprint does not have a description.json")
                pass
            print(data['name'])
            if data['name'].lower() == search_term.lower():
                file = json.load(open(blueprints[i]+"\\blueprint.json"))
                path = blueprints[i]

        if file == 0:
            return None
        return [file, path] #Returns blueprint file, and the path for the blueprint file


    # @classmethod
    # def from_list_block(cls, list: list):
    #     # return cls.super().__init__(bodies=[{"childs": list}], version=3)
    #     super(Blueprint, cls)

    def addBlock(self, block: "Base_Block") -> None:
        self["bodies"][0]["childs"].append(block)

    def addBlocks(self, blocks: list["Base_Block"]) -> None:
        self["bodies"][0]["childs"] += list(blocks)

    def addArray(self, arr: list[list["Base_Block"]]):
        for list in arr:
            self.addBlocks(list)

    def saveToFile(self, file: str):
        """Save the blueprint to a file/path to a file (it overrides any 'blueprint.json' that existed before saving it in that directory)"""
        save = open(file, "w")
        save.write(json.dumps(self, sort_keys=True))
        save.close()

    def getBlocks(self, filter: Callable[..., bool] | None = None):
        """never tested"""
        out = []
        if not callable(filter):
            for dict in self["bodies"][0]["childs"]:
                if (blockType := shapeId_to_class.get(dict["shapeId"])) != None:
                    out.append(blockType.from_dict(dict))
            return out
        else:
            for dict in self["bodies"][0]["childs"]:
                if (blockType := shapeId_to_class.get(dict["shapeId"])) != None:
                    block = blockType.from_dict(dict)
                    if filter(block):
                        out.append(block)
            return out

class ID_Handler:
    id: int
    def __init__(self) -> None:
        self.id = 0

    def assign_id(self):
        out = self.id
        self.id += 1
        return out

class Base_Block(dict):
    def __init__(self, color: str, pos: list[int], shapeId: str, direction: list[int]) -> None:
        super().__init__(self, color=color, pos=dict(zip("xyz", [int(x) for x in pos])), shapeId=shapeId, xaxis=int(direction[0]), zaxis=int(direction[1]))

    @classmethod
    def from_dict(cls, dict: dict):
        return cls(color=dict["color"], pos=dict["pos"].values(), shapeId=dict["shapeId"], direction=(dict["xaxis"], dict["zaxis"]))

    def get_pos(self):
        return np.array(list(self["pos"].values()), dtype=int)

class Connectable_Block(Base_Block):
    input_count: int
    """Number of incoming connections"""
    output_count: int
    """Number of outgoing connections"""

    def __init__(self, color: str, controllers, id: int | ID_Handler, pos: list[int], shapeId: str, direction: list[int] | None = None) -> None:
        """just realized bounded blocks always have the same orientation"""
        super().__init__(color, pos, shapeId, [1, 3])
        if isinstance(id, int):
            self["controller"] = { "active": False, "controllers": controllers, "id": id, "joints": None }
        elif isinstance(id, ID_Handler):
            self["controller"] = { "active": False, "controllers": controllers, "id": id.assign_id(), "joints": None }
        else:
            raise TypeError(f"id must either be of type int or type ID_Handler (got {type(id)})")
        self.input_count = 0
        self.output_count = 0

    @classmethod
    def from_dict(cls, dict: dict):
        return cls(dict["color"], dict["controller"]["controllers"], dict["controller"]["id"], dict["pos"].values(), dict["shapeId"], (dict["xaxis"], dict["zaxis"]))

    def getId(self):
        """Get id for connecting system"""
        return self["controller"]["id"]

    def connectTo(self, dest: "Connectable_Block") -> None:
        """Connect this Connectable_Block to another Connectable_Block"""
        if self["controller"]["controllers"] == None:
            self["controller"]["controllers"] = []
        if (id := { "id": dest.getId() }) not in self["controller"]["controllers"]:
            self["controller"]["controllers"].append(id)
            self.output_count += 1
            dest.input_count += 1
        if self.output_count > 255:
            raise Exception("outgoing connection limit exceeded")
        if dest.input_count > 255:
            raise Exception("incoming connection limit exceeded")
        # print("in:", self.getId(), self.__class__.__name__, "Output count:", self.output_count)
        # print("out:", dest.getId(), dest.__class__.__name__, "Input count:", dest.input_count)


    def connectToMultiple(self, dest: list["Connectable_Block"]) -> None:
        """Connect this Connectable_Block to many Connectable_Blocks"""
        if self["controller"]["controllers"] == None:
            self["controller"]["controllers"] = []
        for cblock in dest:
            self.connectTo(cblock)
        # self["controller"]["controllers"] += [ { "id": block.getId() } for block in dest ]

class Logic_Gate(Connectable_Block):
    def __init__(self, color: str, controllers, id, mode: str, pos: list[int], direction: list[int]) -> None:
        super().__init__(color, controllers, id, pos, SHAPEID_LOGIC_GATE, direction)
        self["controller"]["mode"] = LOGIC_GATE_MODES[mode]

    @classmethod
    def from_dict(cls, dict: dict):
        return cls(dict["color"], dict["controller"]["controllers"], dict["controller"]["id"], LOGIC_GATE_MODE_TO_NAME[dict["controller"]["mode"]], dict["pos"].values(), (dict["xaxis"], dict["zaxis"]))

class Light_Source(Base_Block):
    def __init__(self, color: str, id: int, pos: list[int], shapeId: str, direction: list[int], coneAngle: int = 0, luminance: int = 50) -> None:
        super().__init__(color, pos, shapeId, direction)
        self["controller"] = { "color": color, "coneAngle": coneAngle, "controllers": None, "id": id, "joints": None, "luminance": luminance }

    @classmethod
    def from_dict(cls, dict: dict):
        return cls(dict["color"], dict["controller"]["id"], dict["pos"].values(), dict["shapeId"], (dict["xaxis"], dict["zaxis"]), dict["controller"]["coneAngle"], dict["controller"]["luminance"])

    def getId(self):
        """Get id for connecting system"""
        return self["controller"]["id"]

class Boundable_Block(Base_Block):
    def __init__(self, bounds: list[int], color: str, pos: list[int], shapeId: str, direction: list[int]) -> None:
        super().__init__(color, pos, shapeId, direction)
        self["bounds"] = { axis: val for axis, val in zip("xyz", bounds) }

    @classmethod
    def from_dict(cls, dict: dict):
        return cls(dict["bounds"], dict["color"], dict["pos"].values(), dict["shapeId"], (dict["xaxis"], dict["zaxis"]))

class Timer(Connectable_Block):
    def __init__(self, color: str, controllers, id, seconds: int, ticks: int, pos: list[int], direction: list[int]) -> None:
        super().__init__(color, controllers, id, pos, SHAPEID_TIMER, direction)
        self["controller"]["seconds"] = seconds
        self["controller"]["ticks"] = ticks

    @classmethod
    def from_dict(cls, dict: dict):
        return cls(dict["color"], dict["controller"]["controllers"], dict["controller"]["id"], dict["controller"]["seconds"], dict["controller"]["ticks"], dict["pos"].values(), (dict["xaxis"], dict["zaxis"]))

class Duck_Holder(Base_Block):
    def __init__(self, color: str, pos: list[int], direction: list[int]) -> None:
        super().__init__(color, pos, SHAPEID_DUCK_HOLDER, direction)

    @classmethod
    def from_dict(cls, dict: dict):
        return cls(dict["color"], dict["pos"].values(), (dict["xaxis"], dict["zaxis"]))

class Headlight(Light_Source):
    def __init__(self, color: str, id: int, pos: list[int], direction: list[int], coneAngle: int = 0, luminance: int = 50) -> None:
        super().__init__(color, id, pos, SHAPEID_HEADLIGHT, direction, coneAngle, luminance)

    @classmethod
    def from_dict(cls, dict: dict):
        return cls(dict["color"], dict["controller"]["id"], dict["pos"].values(), (dict["xaxis"], dict["zaxis"]), dict["controller"]["coneAngle"], dict["controller"]["luminance"])

class Switch(Connectable_Block):
    def __init__(self, color: str, controllers, id: int | ID_Handler, pos: list[int], direction: list[int]) -> None:
        super().__init__(color, controllers, id, pos, SHAPEID_SWITCH, direction)

    @classmethod
    def from_dict(cls, dict: dict):
        return cls(dict["color"], dict["controller"]["controllers"], dict["controller"]["id"], dict["pos"].values(), (dict["xaxis"], dict["zaxis"]))

class Button(Connectable_Block):
    def __init__(self, color: str, controllers, id: int | ID_Handler, pos: list[int], direction: list[int]) -> None:
        super().__init__(color, controllers, id, pos, SHAPEID_BUTTON, direction)

    @classmethod
    def from_dict(cls, dict: dict):
        return cls(dict["color"], dict["controller"]["controllers"], dict["controller"]["id"], dict["pos"].values(), (dict["xaxis"], dict["zaxis"]))

class Glass_Block(Boundable_Block):
    def __init__(self, bounds: list[int], color: str, pos: list[int]) -> None:
        super().__init__(bounds, color, pos, SHAPEID_GLASS_BLOCK, (1, 3))

    @classmethod
    def from_dict(cls, dict: dict):
        return cls(dict["bounds"], dict["color"], dict["pos"].values(), (dict["xaxis"], dict["zaxis"]))

class Thruster(Connectable_Block):
    def __init__(self, color: str, controllers, id: int, level: int, pos: list[int], direction: list[int]) -> None:
        super().__init__(color, controllers, id, pos, SHAPEID_THRUSTER, direction)
        self["controller"].update({"level": level})

    @classmethod
    def from_dict(cls, dict: dict):
        return cls(dict["color"], dict["controller"]["controllers"], dict["controller"]["id"], dict["controller"]["level"], dict["pos"].values(), (dict["xaxis"], dict["zaxis"]))

class Metal_Block_1(Boundable_Block):
    def __init__(self, bounds: list[int], color: str, pos: list[int]) -> None:
        super().__init__(bounds, color, pos, SHAPEID_METAL_BLOCK_1, (1, 3))

    @classmethod
    def from_dict(cls, dict: dict):
        return cls(dict["bounds"].values(), dict["color"], dict["pos"].values(), (dict["xaxis"], dict["zaxis"]))

class Barrier_Block(Boundable_Block):
    def __init__(self, bounds: list[int], color: str, pos: list[int]) -> None:
        super().__init__(bounds, color, pos, SHAPEID_BARRIER_BLOCK, (1, 3))

    @classmethod
    def from_dict(cls, dict: dict):
        return super().from_dict(dict)

class Plastic_Block(Boundable_Block):
    def __init__(self, bounds: list[int], color: str, pos: list[int]) -> None:
        super().__init__(bounds, color, pos, SHAPEID_PLASTIC_BLOCK, (1, 3))

    @classmethod
    def from_dict(cls, dict: dict):
        return super().from_dict(dict)

def decoder(id_handler: ID_Handler, nBits: int, pos: tuple[int, int, int] = (0, 0, 0)):
    """Creates a binary decoder

    `id_handler`: ID_Handler For setting unique ID's for every logic gate.

    `nBits`: int The number of bits the decoder is going to be, the number of possible outputs will be 2 ** nBits.

    `pos`: Tuple[int, int, int] The position where the creation will be located.

    It returns a tuple of lists containing all logic gates used to create the decoder, it is ordered in the following way:

    `inputInterface` (size: nBits), `input` (size: nBits), `input_neg` (size: nBits), `output` (size: 2 ** nBits).
    """

    assert isinstance(id_handler, ID_Handler), "id_handler must be of type %s (got %s)" % (ID_Handler.__name__, type(id_handler).__name__)

    assert isinstance(nBits, int)   , "Number of bits must be an int (got %s)" % type(nBits).__name__
    assert nBits > 0                , "Number of bits must be at least 1 (got %d)" % nBits

    assert isinstance(pos, Iterable), "Position creation must be an iterable of ints (got %s)" % type(pos).__name__
    assert len(pos) > 2             , "Position creation must have length of 3 (got length: %d)" % len(pos)
    assert isinstance(pos[0], int) and isinstance(pos[1], int) and isinstance(pos[2], int), "Position creation must have ints in its indices (got %s)" % str(pos)

    if nBits >= 8:
        warnings.warn("Decoder creation may not work due to Scrap Mechanic's 255 connections per gate limit", RuntimeWarning)

    inputInterface: list[Logic_Gate] = []
    input:          list[Logic_Gate] = []
    input_neg:      list[Logic_Gate] = []
    output:         list[Logic_Gate] = []

    # Input bits
    for b in range(nBits):
        inputInterface.append(Logic_Gate("ff0000", [], id_handler.assign_id(), "or", (b + pos[0], 0 + pos[1], 0 + pos[2]), DIRECTION_PLUSX))

    # Body
    for b in range(nBits):
        input.append(Logic_Gate("000000", [], id_handler.assign_id(), "and", (b + pos[0], 1 + pos[1], 0 + pos[2]), DIRECTION_PLUSX))
        input_neg.append(Logic_Gate("000000", [], id_handler.assign_id(), "nand", (b + pos[0], 2 + pos[1], 0 + pos[2]), DIRECTION_PLUSX))
        inputInterface[b].connectToMultiple((input[b], input_neg[b]))

    # Output
    for n in range(2 ** nBits):
        # output.append(Logic_Gate("0000ff", [], id_handler.assign_id(), "and", (-1 - n // (nBits * 2) + pos[0], n % (nBits * 2) + pos[1], 0 + pos[2]), DIRECTION_PLUSX)) # square shape
        output.append(Logic_Gate("0000ff", [], id_handler.assign_id(), "and", (-1 + pos[0], n + 1 + pos[1], 0 + pos[2]), DIRECTION_PLUSX)) # line shape
        combination = ("{0:0>" + f"{nBits}" + "}").format(bin(n)[2:])
        for i, b in enumerate(combination):
            if b == "1":
                input[i].connectTo(output[n])
            else:
                input_neg[i].connectTo(output[n])

    return inputInterface, input, input_neg, output

def programmable_timer(id_handler: ID_Handler, nBits: int, pos: tuple[int, int, int] = (0, 0, 0)):
    """Creates a programmable timer

    returns:
        `decoderParts`, `or0`, `and0`, `and1`, `[orIn]`, `[orOut]`"""
    or0: list[Logic_Gate] = []
    and0: list[Logic_Gate] = []
    and1: list[Logic_Gate] = []
    dec = decoder(id_handler, nBits, pos)
    orIn: Logic_Gate = Logic_Gate("ff0000", [], id_handler.assign_id(), "or", (-3 + pos[0], 0 + pos[1], 0 + pos[2]), DIRECTION_PLUSX)
    orOut: Logic_Gate = Logic_Gate("0000ff", [], id_handler.assign_id(), "or", (-4 + pos[0], 0 + pos[1], 0 + pos[2]), DIRECTION_PLUSX)
    for n in range(2 ** nBits):
        or0.append(Logic_Gate("000000", [], id_handler.assign_id(), "or", (-2 + pos[0], n + 1 + pos[1], 0 + pos[2]), DIRECTION_PLUSX))
        dec[-1][n].connectToMultiple(or0[:n + 1])
        and0.append(Logic_Gate("000000", [], id_handler.assign_id(), "and", (-3 + pos[0], n + 1 + pos[1], 0 + pos[2]), DIRECTION_PLUSX))
        or0[n].connectTo(and0[n])
        if n != 0: and0[n - 1].connectTo(and0[n])
        and1.append(Logic_Gate("000000", [], id_handler.assign_id(), "and", (-4 + pos[0], n + 1 + pos[1], 0 + pos[2]), DIRECTION_PLUSX))
        and0[n].connectTo(and1[n])
        dec[-1][n].connectTo(and1[n])
        and1[n].connectTo(orOut)
    orIn.connectTo(and0[0])
    decoderParts = []
    for part in dec:
        decoderParts += part
    return decoderParts, or0, and0, and1, [orIn], [orOut]

def register(id_handler: ID_Handler, bitLength: int, enableGate: Logic_Gate | None = None, pos: tuple[int, int, int] = (0, 0, 0)):
    """Returns input_xors, write_ands, [enableGate], selfwired_xors"""
    selfwired_xors: list[Logic_Gate] = []
    write_ands: list[Logic_Gate] = []
    input_xors: list[Logic_Gate] = []
    doGotEnableGate: bool = False

    if not enableGate:
        enableGate = Logic_Gate("ff0000", [], id_handler.assign_id(), "or", (bitLength + pos[0], 0 + pos[1], 1 + pos[2]), DIRECTION_PLUSX)
    else:
        doGotEnableGate = True

    for b in range(bitLength):
        input_xors.append(Logic_Gate("ff0000", [], id_handler.assign_id(), "xor", (b + pos[0], 0 + pos[1], 0 + pos[2]), DIRECTION_PLUSX))
        write_ands.append(Logic_Gate("000000", [], id_handler.assign_id(), "and", (b + pos[0], 0 + pos[1], 1 + pos[2]), DIRECTION_PLUSX))
        input_xors[b].connectTo(write_ands[b])
        selfwired_xors.append(Logic_Gate("0000ff", [], id_handler.assign_id(), "xor", (b + pos[0], 0 + pos[1], 2 + pos[2]), DIRECTION_PLUSX))
        selfwired_xors[b].connectTo(selfwired_xors[b])
        write_ands[b].connectTo(selfwired_xors[b])
        selfwired_xors[b].connectTo(input_xors[b])
    enableGate.connectToMultiple(write_ands)
    if doGotEnableGate:
        return input_xors, write_ands, selfwired_xors
    else:
        return input_xors, write_ands, [enableGate], selfwired_xors

def cla_1tick(id_handler: ID_Handler, n_bit: int, pos=np.array([0, 0, 0], dtype=int)):
    """
    returns (
        0: [add_subtract],

        1: [enable_out],

        2: input_a,

        3: [Barrier_Block((1, 1, 1), black, (n_bit, 0, 0), DIRECTION_PLUSX)],

        4: input_b,



        5: and_a1,

        6: xor_b1,



        7: output,

        8: [l for l in and_matrix_b2.flatten() if isinstance(l, Logic_Gate)],

        9: or_b3,

        10: [carry_inverted],



        11: enable_a4,


        12: [timer_carry_in_to_matrix],

        13: [timer_output_0],

        14: timers_xor_b1_to_output,

        15: timers_and_a1_to_or_b3

        )"""
    pos = np.array(pos, dtype=int)
    red = "ff0000"
    black = "000000"
    blue = "0000ff"

    assert n_bit > 0
    input_a = [Logic_Gate(red, [], id_handler.assign_id(), "or", pos + (x, 0, 0), DIRECTION_PLUSX) for x in range(n_bit)]
    input_b = [Logic_Gate(red, [], id_handler.assign_id(), "xor", pos + (x + n_bit + 1, 0, 0), DIRECTION_PLUSX) for x in range(n_bit)]

    and_a1 = [Logic_Gate(black, [], id_handler.assign_id(), "and", pos + (x, 1, 0), DIRECTION_PLUSX) for x in range(n_bit)]
    xor_b1 = [Logic_Gate(black, [], id_handler.assign_id(), "xor", pos + (x + n_bit + 1, 1, 0), DIRECTION_PLUSX) for x in range(n_bit)]

    output = [Logic_Gate(blue, [], id_handler.assign_id(), "xor", pos + (x, 2, 0), DIRECTION_PLUSX) for x in range(n_bit)]
    and_matrix_b2 = np.full((n_bit, n_bit), None, dtype=Logic_Gate)
    for n in range(n_bit):
        and_matrix_b2[n, : n + 1] = [Logic_Gate(black, [], id_handler.assign_id(), "and", pos + (x + n_bit + 1, 2, n_bit - n - 1), DIRECTION_PLUSX) for x in range(n + 1)]

    or_b3 = [Logic_Gate(black if x != 0 else blue, [], id_handler.assign_id(), "or", pos + (x + n_bit + 1, 3, 0), DIRECTION_PLUSX) for x in range(n_bit)]
    carry_inverted = Logic_Gate(blue, [], id_handler.assign_id(), "nor", pos + (n_bit + 1, 4, 0), DIRECTION_PLUSX)

    enable_a4 = [Logic_Gate(blue, [], id_handler.assign_id(), "and", pos + (x, 3, 0), DIRECTION_PLUSX) for x in range(n_bit)]

    add_subtract = Logic_Gate(red, [], id_handler.assign_id(), "or", pos + (n_bit * 2 + 1, 2, 0), DIRECTION_PLUSX)
    enable_out = Logic_Gate(red, [], id_handler.assign_id(), "or", pos + (n_bit * 2 + 1, 3, 0), DIRECTION_PLUSX)


    # add_subtract.connectToMultiple(and_matrix_b2[-1])
    add_subtract.connectTo(output[-1])
    add_subtract.connectToMultiple(input_b)
    enable_out.connectToMultiple(enable_a4)


    # Timing for 1 tick input
    timer_carry_in_to_matrix = Timer(black, [], id_handler.assign_id(), 0, 0, pos + (n_bit * 2 + 1, 2, 1), DIRECTION_PLUSX)
    add_subtract.connectTo(timer_carry_in_to_matrix)
    timer_carry_in_to_matrix.connectToMultiple(and_matrix_b2[-1])

    timer_output_0 = Timer(black, [], id_handler.assign_id(), 0, 2, pos + (n_bit - 1, 2, 1), DIRECTION_PLUSX)
    output[-1].connectTo(timer_output_0)
    timer_output_0.connectTo(enable_a4[-1])

    timers_xor_b1_to_output = [Timer(black, [], id_handler.assign_id(), 0, 1, pos + (n_bit + x + 1, 1, 1), DIRECTION_PLUSX) for x in range(n_bit-1)]
    parallel_connections(xor_b1[ : -1], timers_xor_b1_to_output)
    parallel_connections(timers_xor_b1_to_output, output[ : -1])

    timers_and_a1_to_or_b3 = [Timer(black, [], id_handler.assign_id(), 0, 0, pos + (x, 1, 1), DIRECTION_PLUSX) for x in range(n_bit)]
    parallel_connections(and_a1, timers_and_a1_to_or_b3)
    parallel_connections(timers_and_a1_to_or_b3, or_b3)
    timers_and_a1_to_or_b3[0].connectTo(carry_inverted)
    ##


    parallel_connections(input_a, and_a1)
    parallel_connections(input_a, xor_b1)

    # parallel_connections(and_a1[1 : ], output[ : -1])#bad
    # and_a1[0].connectTo(or_b3[0])#bad
    # parallel_connections(and_a1, or_b3)
    for n in range(1, len(and_a1)):
        and_a1[n].connectToMultiple(l for l in and_matrix_b2[n - 1, : n] if isinstance(l, Logic_Gate))
    # parallel_connections(output, enable_a4)
    parallel_connections(output[ : -1], enable_a4[ : -1])


    parallel_connections(input_b, and_a1)
    parallel_connections(input_b, xor_b1)


    input_a[-1].connectTo(output[-1])
    input_b[-1].connectTo(output[-1])
    # parallel_connections(xor_b1[ : -1], output[ : -1]) #
    # parallel_connections(input_a, output)
    # parallel_connections(input_b, output)
    for n in range(n_bit):
        xor_b1[n].connectToMultiple(l for l in and_matrix_b2[n : , : n + 1].flatten() if isinstance(l, Logic_Gate))
    for n in range(n_bit):
        [l.connectTo(or_b3[n]) for l in and_matrix_b2[n : , n].flatten() if isinstance(l, Logic_Gate)]
    [l.connectTo(carry_inverted) for l in and_matrix_b2[0 : , 0].flatten() if isinstance(l, Logic_Gate)]


    parallel_connections(or_b3[1 : ], output[ : -1])

    return  ([add_subtract],
            [enable_out],
            input_a,
            [Barrier_Block((1, 1, 1), black, pos + (n_bit, 0, 0))],
            input_b,

            and_a1,
            xor_b1,

            output,
            [l for l in and_matrix_b2.flatten() if isinstance(l, Logic_Gate)],
            or_b3,
            [carry_inverted],

            enable_a4,

            [timer_carry_in_to_matrix],
            [timer_output_0],
            timers_xor_b1_to_output,
            timers_and_a1_to_or_b3)

def n_bit_m_right_shift(id_handler: ID_Handler, nbit: int, nbit_shift: int, pos: np.ndarray | tuple[int, int, int]):
    """
    returns (

        input,

        input_shift,

        inpt_shft,

        not_inpt_shft,

        mat_muxs.flatten().tolist(),

        [always_off]

    )
    """
    def mux2x1(id_handler: ID_Handler, select: Logic_Gate, not_select: Logic_Gate, pos: np.ndarray | tuple[int, int, int]):
        """returns (input_0, input_1, output)"""

        pos = np.array(pos, dtype=int)
        input_0 = Logic_Gate("ff0000", [], id_handler, "and", pos + (0, 0, 0), (2, 1))
        input_1 = Logic_Gate("ff0000", [], id_handler, "and", pos + (0, 0, 1), (2, 1))
        output = Logic_Gate("0000ff", [], id_handler, "or", pos + (0, 1, 0), (2, 1))
        not_select.connectTo(input_0)
        select.connectTo(input_1)
        input_0.connectTo(output)
        input_1.connectTo(output)

        return input_0, input_1, output
    pos = np.array(pos, dtype=int)
    input = [Logic_Gate("ff0000", [], id_handler, "or", pos + (x, 0, 0), (2, 1)) for x in range(nbit)]
    input_shift = [Logic_Gate("ff0000", [], id_handler, "or", pos + (-nbit_shift - 2 + x + 1, 0, 0), (2, 1)) for x in range(nbit_shift)]

    inpt_shft = [Logic_Gate("000000", [], id_handler, "and", pos + (-nbit_shift - 2 + x + 1, 1, 0), (2, 1)) for x in range(nbit_shift)]
    parallel_connections(input_shift, inpt_shft)
    not_inpt_shft = [Logic_Gate("000000", [], id_handler, "nand", pos + (-nbit_shift - 2 + x + 1, 2, 0), (2, 1)) for x in range(nbit_shift)]
    parallel_connections(input_shift, not_inpt_shft)

    mat_muxs = []
    for n, (selector, not_selector) in enumerate(zip(inpt_shft, not_inpt_shft)):
        muxs = [mux2x1(id_handler, selector, not_selector, pos + (m, 1 + n * 2, 0)) for m in range(nbit)]
        mat_muxs.append(muxs)
    mat_muxs = np.array(mat_muxs)
    parallel_connections(input, mat_muxs[0, 2**(nbit_shift - 1):, 1])
    parallel_connections(input, mat_muxs[0, :, 0])

    always_off = Logic_Gate("000000", [], id_handler, "and", pos + (-1, 1, 0), (2, 1))
    always_off.connectTo(always_off)
    for n in range(nbit_shift):
        if n < nbit_shift - 1:
            parallel_connections(mat_muxs[n, :, 2], mat_muxs[n + 1, :                   , 0])
            parallel_connections(mat_muxs[n, :, 2], mat_muxs[n + 1, 2 ** (nbit_shift - n - 2): , 1])

        always_off.connectToMultiple(mat_muxs[n, :2 ** (nbit_shift - n - 1) , 1])

    return (
        input,
        input_shift,
        inpt_shft,
        not_inpt_shft,
        mat_muxs.flatten().tolist(),
        [always_off]
    )

def n_bit_comparator(id_handler, n_bit, pos):
    """
    returns (
        0: input_a,

        1: input_b,

        2: and0_a,

        3: nand1_a,

        4: and2_a,

        5: and0_b,

        6: xnor1_b,

        7: and2_b,

        8: [output]
    )"""
    pos = np.array(pos, dtype=int)
    input_a = [Logic_Gate("ff0000", [], id_handler, "or", pos + (x, 0, 0), (2, 1)) for x in range(n_bit)]
    input_b = [Logic_Gate("ff0000", [], id_handler, "xor", pos + (x + n_bit + 1, 0, 0), (2, 1)) for x in range(n_bit)]

    and0_a = [Logic_Gate("000000", [], id_handler, "and", pos + (x, 1, 0), (2, 1)) for x in range(n_bit)]
    nand1_a = [Logic_Gate("000000", [], id_handler, "nand", pos + (x, 2, 0), (2, 1)) for x in range(n_bit)]
    and2_a = [Logic_Gate("000000", [], id_handler, "and", pos + (x, 3, 0), (2, 1)) for x in range(n_bit)]

    and0_b = [Logic_Gate("000000", [], id_handler, "and", pos + (n_bit + x + 1, 1, 0), (2, 1)) for x in range(n_bit)]
    xnor1_b = [Logic_Gate("000000", [], id_handler, "xnor", pos + (n_bit + x + 1, 2, 0), (2, 1)) for x in range(n_bit)]
    and2_b = [Logic_Gate("000000", [], id_handler, "and", pos + (n_bit + x + 1, 3, 0), (2, 1)) for x in range(n_bit)]

    output = Logic_Gate("0000ff", [], id_handler, "or", pos + (n_bit, 3, 0), (2, 1))

    parallel_connections(input_a, and0_a)
    parallel_connections(input_a, nand1_a)
    parallel_connections(nand1_a, and2_a)
    parallel_connections(and0_a, xnor1_b)
    parallel_connections(and2_a[1:], and2_b[:-1])
    and2_a[0].connectTo(output)

    parallel_connections(input_b, and0_b)
    parallel_connections(and0_b, xnor1_b)
    for n in range(len(xnor1_b)):
        xnor1_b[n].connectToMultiple(and2_b[n:])
    parallel_connections(and0_b, and2_a)
    for lg in and2_b: lg.connectTo(output)

    return input_a, input_b, and0_a, nand1_a, and2_a, and0_b, xnor1_b, and2_b, [output]

def parallel_connections(a: list[Logic_Gate], b: list[Logic_Gate]) -> None:
    for l0, l1 in zip(a, b): l0.connectTo(l1)

def to_list_bit(num: int, size: int):
    return [(num >> n) & 1 for n in range(size)][::-1]

def count_bits(num):
    """counts the number of 1's in the binary number (positive)"""
    result = 0
    while True:
        if num == 0: break
        if num & 1: result += 1
        num >>= 1
    return result

def xor_memory(
        id_handler: ID_Handler,
        pos: list[int, int, int] | np.ndarray,
        nbit_input: int, nbit_output: int,
        truth_table: list[int],
        inputs: list[Logic_Gate] | None = None, outputs: list[Logic_Gate] | None = None,
        order: Literal["1d", "inBackOutFront", "4x1xn", "ands below xors"] = "inBackOutFront",
        doGlitchWeld=False,
        xbasis=np.array([1,0,0], dtype=int), ybasis=np.array([0,1,0], dtype=int), zbasis=np.array([0,0,1], dtype=int)
    ):
    """Returns inputs, outputs, ands"""
    provided_inputs, provided_outputs = bool(inputs), bool(outputs)
    pos = np.array(pos, dtype=int)
    if xbasis is not None:
        xbasis = np.array(xbasis, dtype=int)
    if ybasis is not None:
        ybasis = np.array(ybasis, dtype=int)
    if zbasis is not None:
        zbasis = np.array(zbasis, dtype=int)

    if inputs is None:
        if order == "1d":
            inputs = [Logic_Gate("ff0000", [], id_handler, "or", xbasis * n + pos, (2, 1)) for n in range(nbit_input)]
        elif order == "inBackOutFront":
            inputs = [Logic_Gate("ff0000", [], id_handler, "or", xbasis * n + pos, (2, 1)) for n in range(nbit_input)]
        elif order == "4x1xn":
            inputs = [Logic_Gate("ff0000", [], id_handler, "or", zbasis * n + 2 * ybasis + pos, (3, -2)) for n in range(nbit_input)]
        else:
            inputs = [Logic_Gate("ff0000", [], id_handler, "or", xbasis * n + pos, (2, 1)) for n in range(nbit_input)]
    if outputs is None:
        if order == "1d":
            outputs = [Logic_Gate("0000ff", [], id_handler, "xor", xbasis * (n + nbit_input) + pos, (2, 1)) for n in range(nbit_output)]
        elif order == "inBackOutFront":
            outputs = [Logic_Gate("0000ff", [], id_handler, "xor", xbasis * n + ybasis * 2 + pos, (2, 1)) for n in range(nbit_output)]
        elif order == "4x1xn":
            outputs = [Logic_Gate("0000ff", [], id_handler, "xor", xbasis * nbit_input + ybasis + zbasis * n + pos, (3, 2)) for n in range(nbit_output)]
        elif order == "ands below xors":
            outputs = [Logic_Gate("0000ff", [], id_handler, "xor", xbasis * n + ybasis + pos, (2, 1)) for n in range(nbit_output)]
        else:
            # outputs = [Logic_Gate("0000ff", [], id_handler, "xor", xbasis * n + ybasis * 2 + pos, (2, 1)) for n in range(nbit_output)]
            outputs = [Logic_Gate("0000ff", [], id_handler, "xor", xbasis * n + ybasis * 3 + pos, (2, 1)) for n in range(nbit_output)]
    if doGlitchWeld:
        xbasis = zbasis = ybasis = np.array((0,0,0), dtype=int)

    def new_pos_dir(_order):
        if _order == "1d":
            return xbasis * (len(ands) + nbit_output + nbit_input) + pos                                                    , (2, 1)
        elif _order == "inBackOutFront":
            return xbasis * (nbit_input - 1 - (len(ands) % nbit_input)) + ybasis + zbasis * (len(ands) // nbit_input) + pos , (2, 1)
        elif _order == "4x1xn":
            return xbasis * (len(ands) % nbit_input) + ybasis * 2 + zbasis * (len(ands) // nbit_input) + pos                , (1, -2)
        elif _order == "ands below xors":
            return xbasis * (len(ands) % nbit_input) + ybasis + zbasis * ((len(ands) // nbit_input) - 1) + pos              , (2, 1)
        else:
            # return xbasis * (len(ands) % nbit_input) + ybasis + zbasis * (len(ands) // nbit_input) + pos                    , (2, 1)
            return xbasis * (len(ands) % 7) + ybasis * (((len(ands) // 7) % 2) + 1) + zbasis * (len(ands) // 14) + pos                    , (2, 1)

    xors_keys = []
    def evaluate(x):
        result = 0
        for input, output in xors_keys:
            if input == -1 and x == 0:
                result = output
                continue
            if (input & x) == input:
                result ^= output
        return result

    ands: list[Logic_Gate] = []
    for input, output in enumerate(truth_table):
        if output == -1: continue
        current_output = evaluate(input)
        if current_output != output:
            difference = current_output ^ output

            def connect_many_to_one_if_state(state: int, list_states: list, gates_src: list[Logic_Gate], gate_dest: Logic_Gate):
                for i, b in enumerate(list_states):
                    if state == -1: gates_src[i].connectTo(gate_dest)
                    elif b == state: gates_src[i].connectTo(gate_dest)

            def connect_one_to_many_if_state(state: int, list_states: list, gate_src: Logic_Gate, gates_dest: list[Logic_Gate]):
                for i, b in enumerate(list_states):
                    if state == -1: gate_src.connectTo(gates_dest[i])
                    elif b == state: gate_src.connectTo(gates_dest[i])

            if count_bits(input) != 1:
                if input != 0:
                    new_gate =  Logic_Gate("000000", [], id_handler, "and", *new_pos_dir(order))
                    connect_many_to_one_if_state(1, to_list_bit(input, nbit_input), inputs, new_gate)
                    connect_one_to_many_if_state(1, to_list_bit(difference, nbit_output), new_gate, outputs)
                    ands.append(new_gate)
                else:
                    new_gate =  Logic_Gate("000000", [], id_handler, "nor", *new_pos_dir(order))
                    connect_many_to_one_if_state(-1, to_list_bit(input, nbit_input), inputs, new_gate)
                    connect_one_to_many_if_state(1, to_list_bit(difference, nbit_output), new_gate, outputs)
                    xors_keys.append((-1, output))
                    ands.append(new_gate)
                    continue
            else:
                idx_input_on = [idx for idx, b in enumerate(to_list_bit(input, nbit_input)) if b][0]
                connect_one_to_many_if_state(1, to_list_bit(difference, nbit_output), inputs[idx_input_on], outputs)

            xors_keys.append((input, difference))
    print(f"{len(ands) = }")
    return inputs if not provided_inputs else [], outputs if not provided_outputs else [], ands

def MakeFakeConnect(cls: type):
    class FakeConnect(cls):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            if not self["controller"]["controllers"]:
                self["controller"]["controllers"] = []
            self.fake_connect_list = []

        def connectTo(self, other):
            self.fake_connect_list.append(other)

        def connectToMultiple(self, list_block: list):
            for block in list_block:
                self.connectTo(block)

        def wire(self, id_handler, bp):
            if len(self.fake_connect_list) <= 255:  # Direct wiring
                for c in self.fake_connect_list:
                    super(cls, self).connectTo(c)
            else:   # Intermediate gates
                n_intermediate_gates = int(np.ceil(len(self.fake_connect_list) / 255))
                pos = self.get_pos()
                intermediate_gates = [Logic_Gate("000000", [], id_handler, "or", pos + (0,0,-1-n), (2, 1)) for n in range(n_intermediate_gates)]
                [super(cls, self).connectTo(c) for c in intermediate_gates]
                for gate, chunk in zip(intermediate_gates, np.array_split(self.fake_connect_list, n_intermediate_gates)):
                    gate.connectToMultiple(chunk)
                bp.addBlocks(intermediate_gates)
                print(n_intermediate_gates)
    return FakeConnect

shapeId_to_class = {
    SHAPEID_DUCK_HOLDER: Duck_Holder,
    SHAPEID_LOGIC_GATE: Logic_Gate,
    SHAPEID_TIMER: Timer,
    SHAPEID_HEADLIGHT: Headlight,
    SHAPEID_SWITCH: Switch,
    SHAPEID_BUTTON: Button,
    SHAPEID_GLASS_BLOCK: Glass_Block,
    SHAPEID_THRUSTER: Thruster,
    SHAPEID_METAL_BLOCK_1: Metal_Block_1,
    SHAPEID_BARRIER_BLOCK: Barrier_Block,
    SHAPEID_PLASTIC_BLOCK: Plastic_Block,
    SHAPEID_SENSOR: Sensor,
}
