# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
###############################################################################

import sys
import onnx
from .optimizer import LinkedNode, Solution


def remove_cast(lnodes, op_set):
    while True:
        sln = []
        for n_ in lnodes:
            if n_.op_type in op_set and n_.in_single_path:
                if n_.precedence[0].op_type == 'Cast' and n_.successor[0].op_type == 'Cast':
                    sln.append(Solution(None, n_.precedence[0], n_.precedence[0], n_))
                    sln.append(Solution(n_, n_.successor[0], n_.successor[0], None))
                    break

        if len(sln) == 0:
            break

        for s_ in sln:
            lnodes = s_.apply(lnodes)[0]

    return lnodes


def decast(origin_model, oplist):
    """
    remove the ONNX cast op from the specified operator.
    :param origin_model:these
    :param oplist:
    :return:
    """
    graph = origin_model.graph
    nodelist = list(graph.node)
    del graph.node[:]

    all_nodes = LinkedNode.build_from_onnx(nodelist,
                                           [],
                                           [i_.name for i_ in graph.input],
                                           [o_.name for o_ in graph.output])

    nodes = remove_cast(all_nodes, set(oplist))
    for n_ in nodes:
        graph.node.extend(n_.generate())

    return origin_model


def main():
    if len(sys.argv) < 4:
        print('decast.py model_in  model_out <op1, ...>')
        return

    input = sys.argv[1]
    output = sys.argv[2]
    op_list = sys.argv[3:]

    oxml = onnx.load_model(input)
    oxml = decast(oxml, op_list)
    onnx.save_model(oxml, output)


if __name__ == "__main__":
    main()
