/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {
std::function<void(OpSchema&)> RNNDocGeneratorOld(const char* /*name*/) {
  return [=](OpSchema& schema) {
    schema.Attr(
        "direction",
        "Specify if the RNN is forward, reverse, or bidirectional. "
        "Must be one of forward (default), reverse, or bidirectional.",
        AttributeProto::STRING,
        std::string("foward"));
    schema.Attr("hidden_size", "Number of neurons in the hidden layer", AttributeProto::INT, OPTIONAL_VALUE);
    schema.Attr(
        "activation_alpha",
        "Optional scaling values used by some activation functions. The values "
        "are consumed in the order of activation functions, for example (f, g, h) "
        "in LSTM.",
        AttributeProto::FLOATS,
        OPTIONAL_VALUE);
    schema.Attr(
        "activation_beta",
        "Optional scaling values used by some activation functions. The values "
        "are consumed in the order of activation functions, for example (f, g, h) "
        "in LSTM.",
        AttributeProto::FLOATS,
        OPTIONAL_VALUE);
    schema.Attr(
        "output_sequence",
        "The sequence output for the hidden is optional if 0. Default 0.",
        AttributeProto::INT,
        static_cast<int64_t>(0));
    schema.Attr(
        "clip",
        "Cell clip threshold. Clipping bounds the elements of a tensor "
        "in the range of [-threshold, +threshold] and is applied to the input "
        "of activations. No clip if not specified.",
        AttributeProto::FLOAT,
        OPTIONAL_VALUE);
    schema.Input(
        0,
        "X",
        "The input sequences packed (and potentially padded) into one 3-D "
        "tensor with the shape of `[seq_length, batch_size, input_size]`.",
        "T");
    schema.Input(
        4,
        "sequence_lens",
        "Optional tensor specifying lengths of the sequences in a batch. "
        "If not specified - assumed all sequences in the batch to have "
        "length `seq_length`. It has shape `[batch_size]`.",
        "T1",
        OpSchema::Optional);
    schema.Input(
        5,
        "initial_h",
        "Optional initial value of the hidden. If not specified - assumed "
        "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.",
        "T",
        OpSchema::Optional);
    schema.Output(
        0,
        "Y",
        "A tensor that concats all the intermediate output values of the hidden. "
        "It has shape `[seq_length, num_directions, batch_size, hidden_size]`. "
        "It is optional if `output_sequence` is 0.",
        "T",
        OpSchema::Optional);
    schema.Output(
        1,
        "Y_h",
        "The last output value of the hidden. It has shape "
        "`[num_directions, batch_size, hidden_size]`.",
        "T");
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.TypeConstraint("T1", {"tensor(int32)"}, "Constrain seq_lens to integer tensor.");
  };
}

static const char* GRU_ver1_doc = R"DOC(
Computes an one-layer GRU. This operator is usually supported via some custom
implementation such as CuDNN.

Notations:

`X` - input tensor

`z` - update gate

`r` - reset gate

`h` - hidden gate

`t` - time step (t-1 means previous time step)

`W[zrh]` - W parameter weight matrix for update, reset, and hidden gates

`R[zrh]` - R recurrence weight matrix for update, reset, and hidden gates

`Wb[zrh]` - W bias vectors for update, reset, and hidden gates

`Rb[zrh]` - R bias vectors for update, reset, and hidden gates

`WB[zrh]` - W parameter weight matrix for backward update, reset, and hidden gates

`RB[zrh]` - R recurrence weight matrix for backward update, reset, and hidden gates

`WBb[zrh]` - W bias vectors for backward update, reset, and hidden gates

`RBb[zrh]` - R bias vectors for backward update, reset, and hidden gates

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh):

  - zt = f(Xt*(Wz^T) + Ht-1*Rz + Wbz + Rbz)

  - rt = f(Xt*(Wr^T) + Ht-1*Rr + Wbr + Rbr)

  - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*Rh + Rbh + Wbh) # default, when linear_before_reset = 0

  - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*Rh + Rbh) + Wbh) # when linear_before_reset != 0

  - Ht = (1 - zt) (.) ht + zt (.) Ht-1
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    GRU,
    1,
    OpSchema()
        .SetDoc(GRU_ver1_doc)
        .Attr(
            "activations",
            "A list of 2 (or 4 if bidirectional) activation functions "
            "for update, reset, and hidden gates. The activation functions must be one "
            "of the activation functions specified above. Optional: See the equations "
            "for default if not specified.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Input(
            1,
            "W",
            "The weight tensor for the gates. Concatenation of `W[zrh]` and `WB[zrh]` "
            "(if bidirectional) along dimension 0. This tensor has shape "
            "`[num_directions, 3*hidden_size, input_size]`.",
            "T")
        .Input(
            2,
            "R",
            "The recurrence weight tensor. Concatenation of `R[zrh]` and `RB[zrh]` "
            "(if bidirectional) along dimension 0. This tensor has shape "
            "`[num_directions, 3*hidden_size, hidden_size]`.",
            "T")
        .Input(
            3,
            "B",
            "The bias tensor for the gates. Concatenation of `[Wb[zrh], Rb[zrh]]` and "
            "`[WBb[zrh], RBb[zrh]]` (if bidirectional) along dimension 0. This tensor "
            "has shape `[num_directions, 6*hidden_size]`. Optional: If not specified "
            "- assumed to be 0",
            "T",
            OpSchema::Optional)
        .FillUsing(RNNDocGeneratorOld("GRU")));

// Versions 1 to 6 of RNN/LSTM and versions 3 to 6 of GRU:

void RNNShapeInference1(InferenceContext& ctx) {
  TensorShapeProto::Dimension num_directions, seq_length, batch_size, hidden_size;

  auto direction = getAttribute(ctx, "direction", "forward");
  if ((direction == "forward") || (direction == "reverse"))
    num_directions.set_dim_value(1);
  else if (direction == "bidirectional")
    num_directions.set_dim_value(2);
  // else leave num_directions unknown in case of incorrect attribute value

  auto hidden_size_value = getAttribute(ctx, "hidden_size", -1);
  if (hidden_size_value > 0)
    hidden_size.set_dim_value(hidden_size_value);

  if (hasInputShape(ctx, 0)) {
    auto& first_input_shape = getInputShape(ctx, 0);
    seq_length = first_input_shape.dim(0);
    batch_size = first_input_shape.dim(1);
  }

  // The treatment of outputs is a bit complicated because of the combination of
  // optional outputs and the output_sequence attribute.
  bool output_sequence = (getAttribute(ctx, "output_sequence", 0) != 0);
  auto num_outputs = ctx.getNumOutputs();

  if (num_outputs == 0)
    return; // Unlikely, but seems legal.

  propagateElemTypeFromInputToOutput(ctx, 0, 0);
  if (num_outputs > 1)
    propagateElemTypeFromInputToOutput(ctx, 0, 1);
  if (num_outputs > 2)
    propagateElemTypeFromInputToOutput(ctx, 0, 2);

  if (output_sequence) {
    // No ambiguity in spec
    updateOutputShape(ctx, 0, {seq_length, num_directions, batch_size, hidden_size}); // Y
    if (num_outputs > 1)
      updateOutputShape(ctx, 1, {num_directions, batch_size, hidden_size}); // Y_h
    if (num_outputs > 2)
      updateOutputShape(ctx, 2, {num_directions, batch_size, hidden_size}); // Y_c
  } else {
    // Documentation suggests that the output Y is absent in this case
    // Different tests seem to disagree on whether Y_h and Y_c, if present,
    // should be in positions 0 & 1 or 1 & 2. updateOutputShape(ctx, 0,
    // {num_directions, batch_size, hidden_size}); // Y_h if (num_outputs > 1)
    // updateOutputShape(ctx, 1, {num_directions, batch_size, hidden_size}); //
    // Y_c
  }
}

std::function<void(OpSchema&)> RNNDocGenerator1(const char* /*name*/) {
  return [=](OpSchema& schema) {
    schema.Attr(
        "direction",
        "Specify if the RNN is forward, reverse, or bidirectional. "
        "Must be one of forward (default), reverse, or bidirectional.",
        AttributeProto::STRING,
        std::string("forward"));
    schema.Attr("hidden_size", "Number of neurons in the hidden layer", AttributeProto::INT, OPTIONAL_VALUE);
    schema.Attr(
        "activation_alpha",
        "Optional scaling values used by some activation functions. The values "
        "are consumed in the order of activation functions, for example (f, g, h) "
        "in LSTM. Default values are the same as of corresponding ONNX operators."
        "For example with LeakyRelu, the default alpha is 0.01.",
        AttributeProto::FLOATS,
        OPTIONAL_VALUE);
    schema.Attr(
        "activation_beta",
        "Optional scaling values used by some activation functions. The values "
        "are consumed in the order of activation functions, for example (f, g, h) "
        "in LSTM. Default values are the same as of corresponding ONNX operators.",
        AttributeProto::FLOATS,
        OPTIONAL_VALUE);
    schema.Attr(
        "output_sequence",
        "The sequence output for the hidden is optional if 0. Default 0.",
        AttributeProto::INT,
        static_cast<int64_t>(0));
    schema.Attr(
        "clip",
        "Cell clip threshold. Clipping bounds the elements of a tensor "
        "in the range of [-threshold, +threshold] and is applied to the input "
        "of activations. No clip if not specified.",
        AttributeProto::FLOAT,
        OPTIONAL_VALUE);
    schema.Input(
        0,
        "X",
        "The input sequences packed (and potentially padded) into one 3-D "
        "tensor with the shape of `[seq_length, batch_size, input_size]`.",
        "T");
    schema.Input(
        4,
        "sequence_lens",
        "Optional tensor specifying lengths of the sequences in a batch. "
        "If not specified - assumed all sequences in the batch to have "
        "length `seq_length`. It has shape `[batch_size]`.",
        "T1",
        OpSchema::Optional);
    schema.Input(
        5,
        "initial_h",
        "Optional initial value of the hidden. If not specified - assumed "
        "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.",
        "T",
        OpSchema::Optional);
    schema.Output(
        0,
        "Y",
        "A tensor that concats all the intermediate output values of the hidden. "
        "It has shape `[seq_length, num_directions, batch_size, hidden_size]`. "
        "It is optional if `output_sequence` is 0.",
        "T",
        OpSchema::Optional);
    schema.Output(
        1,
        "Y_h",
        "The last output value of the hidden. It has shape "
        "`[num_directions, batch_size, hidden_size]`.",
        "T",
        OpSchema::Optional);
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.TypeConstraint("T1", {"tensor(int32)"}, "Constrain seq_lens to integer tensor.");
    schema.TypeAndShapeInferenceFunction(RNNShapeInference1);
  };
}

static const char* RNN_ver1_doc = R"DOC(
Computes an one-layer simple RNN. This operator is usually supported
via some custom implementation such as CuDNN.

Notations:

`X` - input tensor

`i` - input gate

`t` - time step (t-1 means previous time step)

`Wi` - W parameter weight matrix for input gate

`Ri` - R recurrence weight matrix for input gate

`Wbi` - W parameter bias vector for input gate

`Rbi` - R parameter bias vector for input gate

`WBi` - W parameter weight matrix for backward input gate

`RBi` - R recurrence weight matrix for backward input gate

`WBbi` - WR bias vectors for backward input gate

`RBbi` - RR bias vectors for backward input gate

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Tanh):

  - Ht = f(Xt*(Wi^T) + Ht-1*Ri + Wbi + Rbi)
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    RNN,
    1,
    OpSchema()
        .SetDoc(RNN_ver1_doc)
        .Attr(
            "activations",
            "One (or two if bidirectional) activation function for "
            "input gate. The activation function must be one of the activation "
            "functions specified above. Optional: Default `Tanh` if not specified.",
            AttributeProto::STRINGS,
            std::vector<std::string>{"Tanh", "Tanh"})
        .Input(
            1,
            "W",
            "The weight tensor for input gate. Concatenation of `Wi` and `WBi` "
            "(if bidirectional). The tensor has shape "
            "`[num_directions, hidden_size, input_size]`.",
            "T")
        .Input(
            2,
            "R",
            "The recurrence weight tensor. Concatenation of `Ri` and `RBi` "
            "(if bidirectional). The tensor has shape "
            "`[num_directions, hidden_size, hidden_size]`.",
            "T")
        .Input(
            3,
            "B",
            "The bias tensor for input gate. Concatenation of `[Wbi, Rbi]` "
            "and `[WBbi, RBbi]` (if bidirectional). The tensor has shape "
            "`[num_directions, 2*hidden_size]`. Optional: If not specified - assumed "
            "to be 0.",
            "T",
            OpSchema::Optional)
        .FillUsing(RNNDocGenerator1("RNN")));

static const char* GRU_ver3_doc = R"DOC(
Computes an one-layer GRU. This operator is usually supported via some custom
implementation such as CuDNN.

Notations:

`X` - input tensor

`z` - update gate

`r` - reset gate

`h` - hidden gate

`t` - time step (t-1 means previous time step)

`W[zrh]` - W parameter weight matrix for update, reset, and hidden gates

`R[zrh]` - R recurrence weight matrix for update, reset, and hidden gates

`Wb[zrh]` - W bias vectors for update, reset, and hidden gates

`Rb[zrh]` - R bias vectors for update, reset, and hidden gates

`WB[zrh]` - W parameter weight matrix for backward update, reset, and hidden gates

`RB[zrh]` - R recurrence weight matrix for backward update, reset, and hidden gates

`WBb[zrh]` - W bias vectors for backward update, reset, and hidden gates

`RBb[zrh]` - R bias vectors for backward update, reset, and hidden gates

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh):

  - zt = f(Xt*(Wz^T) + Ht-1*Rz + Wbz + Rbz)

  - rt = f(Xt*(Wr^T) + Ht-1*Rr + Wbr + Rbr)

  - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*Rh + Rbh + Wbh) # default, when linear_before_reset = 0

  - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*Rh + Rbh) + Wbh) # when linear_before_reset != 0

  - Ht = (1 - zt) (.) ht + zt (.) Ht-1
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    GRU,
    3,
    OpSchema()
        .SetDoc(GRU_ver3_doc)
        .Attr(
            "activations",
            "A list of 2 (or 4 if bidirectional) activation functions "
            "for update, reset, and hidden gates. The activation functions must be one "
            "of the activation functions specified above. Optional: See the equations "
            "for default if not specified.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "linear_before_reset",
            "When computing the output of the hidden gate, "
            "apply the linear transformation before multiplying by the output of the "
            "reset gate.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(
            1,
            "W",
            "The weight tensor for the gates. Concatenation of `W[zrh]` and `WB[zrh]` "
            "(if bidirectional) along dimension 0. This tensor has shape "
            "`[num_directions, 3*hidden_size, input_size]`.",
            "T")
        .Input(
            2,
            "R",
            "The recurrence weight tensor. Concatenation of `R[zrh]` and `RB[zrh]` "
            "(if bidirectional) along dimension 0. This tensor has shape "
            "`[num_directions, 3*hidden_size, hidden_size]`.",
            "T")
        .Input(
            3,
            "B",
            "The bias tensor for the gates. Concatenation of `[Wb[zrh], Rb[zrh]]` and "
            "`[WBb[zrh], RBb[zrh]]` (if bidirectional) along dimension 0. This tensor "
            "has shape `[num_directions, 6*hidden_size]`. Optional: If not specified "
            "- assumed to be 0",
            "T",
            OpSchema::Optional)
        .FillUsing(RNNDocGenerator1("GRU")));

static const char* LSTM_ver1_doc = R"DOC(
Computes an one-layer LSTM. This operator is usually supported via some
custom implementation such as CuDNN.

Notations:

`X` - input tensor

`i` - input gate

`o` - output gate

`f` - forget gate

`c` - cell gate

`t` - time step (t-1 means previous time step)

`W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates

`R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates

`Wb[iofc]` - W bias vectors for input, output, forget, and cell gates

`Rb[iofc]` - R bias vectors for input, output, forget, and cell gates

`P[iof]`  - P peephole weight vector for input, output, and forget gates

`WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates

`RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates

`WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates

`RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates

`PB[iof]`  - P peephole weight vector for backward input, output, and forget gates

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

  - it = f(Xt*(Wi^T) + Ht-1*Ri + Pi (.) Ct-1 + Wbi + Rbi)

  - ft = f(Xt*(Wf^T) + Ht-1*Rf + Pf (.) Ct-1 + Wbf + Rbf)

  - ct = g(Xt*(Wc^T) + Ht-1*Rc + Wbc + Rbc)

  - Ct = ft (.) Ct-1 + it (.) ct

  - ot = f(Xt*(Wo^T) + Ht-1*Ro + Po (.) Ct + Wbo + Rbo)

  - Ht = ot (.) h(Ct)
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    LSTM,
    1,
    OpSchema()
        .SetDoc(LSTM_ver1_doc)
        .Attr(
            "activations",
            "A list of 3 (or 6 if bidirectional) activation functions "
            "for input, output, forget, cell, and hidden. The activation functions must "
            "be one of the activation functions specified above. Optional: See the equations "
            "for default if not specified.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "input_forget",
            "Couple the input and forget gates if 1, default 0.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(
            1,
            "W",
            "The weight tensor for the gates. Concatenation of `W[iofc]` and "
            "`WB[iofc]` (if bidirectional) along dimension 0. The tensor has shape "
            "`[num_directions, 4*hidden_size, input_size]`.",
            "T")
        .Input(
            2,
            "R",
            "The recurrence weight tensor. Concatenation of `R[iofc]` and "
            "`RB[iofc]` (if bidirectional) along dimension 0. This tensor has shape "
            "`[num_directions, 4*hidden_size, hidden_size]`.",
            "T")
        .Input(
            3,
            "B",
            "The bias tensor for input gate. Concatenation of `[Wb[iofc], Rb[iofc]]`, "
            "and `[WBb[iofc], RBb[iofc]]` (if bidirectional) along dimension 0. This "
            "tensor has shape `[num_directions, 8*hidden_size]`. Optional: If not "
            "specified - assumed to be 0.",
            "T",
            OpSchema::Optional)
        .Input(
            6,
            "initial_c",
            "Optional initial value of the cell. If not specified - assumed "
            "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.",
            "T",
            OpSchema::Optional)
        .Input(
            7,
            "P",
            "The weight tensor for peepholes. Concatenation of `P[iof]` and "
            "`PB[iof]` (if bidirectional) along dimension 0. It has shape "
            "`[num_directions, 3*hidde_size]`. Optional: If not specified - "
            "assumed to be 0.",
            "T",
            OpSchema::Optional)
        .FillUsing(RNNDocGenerator1("LSTM"))
        .Output(
            2,
            "Y_c",
            "The last output value of the cell. It has shape "
            "`[num_directions, batch_size, hidden_size]`.",
            "T",
            OpSchema::Optional));

} // namespace ONNX_NAMESPACE

// Versions 7 to 13 of RNN/LSTM/GRU

namespace ONNX_NAMESPACE {
void RNNShapeInference2(InferenceContext& ctx) {
  TensorShapeProto::Dimension num_directions, seq_length, batch_size, hidden_size;

  auto direction = getAttribute(ctx, "direction", "forward");
  if ((direction == "forward") || (direction == "reverse"))
    num_directions.set_dim_value(1);
  else if (direction == "bidirectional")
    num_directions.set_dim_value(2);
  // else leave num_directions unknown in case of incorrect attribute value

  auto hidden_size_value = getAttribute(ctx, "hidden_size", -1);
  if (hidden_size_value > 0)
    hidden_size.set_dim_value(hidden_size_value);

  if (hasInputShape(ctx, 0)) {
    auto& first_input_shape = getInputShape(ctx, 0);
    if (first_input_shape.dim_size() != 3) {
      fail_shape_inference("First input tensor must have rank 3");
    }
    seq_length = first_input_shape.dim(0);
    batch_size = first_input_shape.dim(1);
  }

  auto num_outputs = ctx.getNumOutputs();

  if (num_outputs > 0) {
    // Y
    propagateElemTypeFromInputToOutput(ctx, 0, 0);
    updateOutputShape(ctx, 0, {seq_length, num_directions, batch_size, hidden_size});
  }

  if (num_outputs > 1) {
    // Y_h
    propagateElemTypeFromInputToOutput(ctx, 0, 1);
    updateOutputShape(ctx, 1, {num_directions, batch_size, hidden_size});
  }

  if (num_outputs > 2) {
    // Y_c : only in the case of LSTM
    propagateElemTypeFromInputToOutput(ctx, 0, 2);
    updateOutputShape(ctx, 2, {num_directions, batch_size, hidden_size});
  }
}

std::function<void(OpSchema&)> RNNDocGenerator2(const char* /*name*/) {
  return [=](OpSchema& schema) {
    schema.Attr(
        "direction",
        "Specify if the RNN is forward, reverse, or bidirectional. "
        "Must be one of forward (default), reverse, or bidirectional.",
        AttributeProto::STRING,
        std::string("forward"));
    schema.Attr("hidden_size", "Number of neurons in the hidden layer", AttributeProto::INT, OPTIONAL_VALUE);
    schema.Attr(
        "activation_alpha",
        "Optional scaling values used by some activation functions. The values "
        "are consumed in the order of activation functions, for example (f, g, h) "
        "in LSTM. Default values are the same as of corresponding ONNX operators."
        "For example with LeakyRelu, the default alpha is 0.01.",
        AttributeProto::FLOATS,
        OPTIONAL_VALUE);
    schema.Attr(
        "activation_beta",
        "Optional scaling values used by some activation functions. The values "
        "are consumed in the order of activation functions, for example (f, g, h) "
        "in LSTM. Default values are the same as of corresponding ONNX operators.",
        AttributeProto::FLOATS,
        OPTIONAL_VALUE);
    schema.Attr(
        "clip",
        "Cell clip threshold. Clipping bounds the elements of a tensor "
        "in the range of [-threshold, +threshold] and is applied to the input "
        "of activations. No clip if not specified.",
        AttributeProto::FLOAT,
        OPTIONAL_VALUE);
    schema.Input(
        0,
        "X",
        "The input sequences packed (and potentially padded) into one 3-D "
        "tensor with the shape of `[seq_length, batch_size, input_size]`.",
        "T");
    schema.Input(
        4,
        "sequence_lens",
        "Optional tensor specifying lengths of the sequences in a batch. "
        "If not specified - assumed all sequences in the batch to have "
        "length `seq_length`. It has shape `[batch_size]`.",
        "T1",
        OpSchema::Optional);
    schema.Input(
        5,
        "initial_h",
        "Optional initial value of the hidden. If not specified - assumed "
        "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.",
        "T",
        OpSchema::Optional);
    schema.Output(
        0,
        "Y",
        "A tensor that concats all the intermediate output values of the hidden. "
        "It has shape `[seq_length, num_directions, batch_size, hidden_size]`. ",
        "T",
        OpSchema::Optional);
    schema.Output(
        1,
        "Y_h",
        "The last output value of the hidden. It has shape "
        "`[num_directions, batch_size, hidden_size]`.",
        "T",
        OpSchema::Optional);
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)"},
        "Constrain input and output types to float tensors.");
    schema.TypeConstraint("T1", {"tensor(int32)"}, "Constrain seq_lens to integer tensor.");
    schema.TypeAndShapeInferenceFunction(RNNShapeInference2);
  };
}

static const char* RNN_ver7_doc = R"DOC(
Computes an one-layer simple RNN. This operator is usually supported
via some custom implementation such as CuDNN.

Notations:

`X` - input tensor

`i` - input gate

`t` - time step (t-1 means previous time step)

`Wi` - W parameter weight matrix for input gate

`Ri` - R recurrence weight matrix for input gate

`Wbi` - W parameter bias vector for input gate

`Rbi` - R parameter bias vector for input gate

`WBi` - W parameter weight matrix for backward input gate

`RBi` - R recurrence weight matrix for backward input gate

`WBbi` - WR bias vectors for backward input gate

`RBbi` - RR bias vectors for backward input gate

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Tanh):

  - Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    RNN,
    7,
    OpSchema()
        .SetDoc(RNN_ver7_doc + GenerateOptionalArgumentsDoc())
        .Attr(
            "activations",
            "One (or two if bidirectional) activation function for "
            "input gate. The activation function must be one of the activation "
            "functions specified above. Optional: Default `Tanh` if not specified.",
            AttributeProto::STRINGS,
            std::vector<std::string>{"Tanh", "Tanh"})
        .Input(
            1,
            "W",
            "The weight tensor for input gate. Concatenation of `Wi` and `WBi` "
            "(if bidirectional). The tensor has shape "
            "`[num_directions, hidden_size, input_size]`.",
            "T")
        .Input(
            2,
            "R",
            "The recurrence weight tensor. Concatenation of `Ri` and `RBi` "
            "(if bidirectional). The tensor has shape "
            "`[num_directions, hidden_size, hidden_size]`.",
            "T")
        .Input(
            3,
            "B",
            "The bias tensor for input gate. Concatenation of `[Wbi, Rbi]` "
            "and `[WBbi, RBbi]` (if bidirectional). The tensor has shape "
            "`[num_directions, 2*hidden_size]`. Optional: If not specified - assumed "
            "to be 0.",
            "T",
            OpSchema::Optional)
        .FillUsing(RNNDocGenerator2("RNN")));

static const char* GRU_ver7_doc = R"DOC(
Computes an one-layer GRU. This operator is usually supported via some custom
implementation such as CuDNN.

Notations:

`X` - input tensor

`z` - update gate

`r` - reset gate

`h` - hidden gate

`t` - time step (t-1 means previous time step)

`W[zrh]` - W parameter weight matrix for update, reset, and hidden gates

`R[zrh]` - R recurrence weight matrix for update, reset, and hidden gates

`Wb[zrh]` - W bias vectors for update, reset, and hidden gates

`Rb[zrh]` - R bias vectors for update, reset, and hidden gates

`WB[zrh]` - W parameter weight matrix for backward update, reset, and hidden gates

`RB[zrh]` - R recurrence weight matrix for backward update, reset, and hidden gates

`WBb[zrh]` - W bias vectors for backward update, reset, and hidden gates

`RBb[zrh]` - R bias vectors for backward update, reset, and hidden gates

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh):

  - zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)

  - rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)

  - ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0

  - ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0

  - Ht = (1 - zt) (.) ht + zt (.) Ht-1
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    GRU,
    7,
    OpSchema()
        .SetDoc(GRU_ver7_doc + GenerateOptionalArgumentsDoc())
        .Attr(
            "activations",
            "A list of 2 (or 4 if bidirectional) activation functions "
            "for update, reset, and hidden gates. The activation functions must be one "
            "of the activation functions specified above. Optional: See the equations "
            "for default if not specified.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "linear_before_reset",
            "When computing the output of the hidden gate, "
            "apply the linear transformation before multiplying by the output of the "
            "reset gate.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(
            1,
            "W",
            "The weight tensor for the gates. Concatenation of `W[zrh]` and `WB[zrh]` "
            "(if bidirectional) along dimension 0. This tensor has shape "
            "`[num_directions, 3*hidden_size, input_size]`.",
            "T")
        .Input(
            2,
            "R",
            "The recurrence weight tensor. Concatenation of `R[zrh]` and `RB[zrh]` "
            "(if bidirectional) along dimension 0. This tensor has shape "
            "`[num_directions, 3*hidden_size, hidden_size]`.",
            "T")
        .Input(
            3,
            "B",
            "The bias tensor for the gates. Concatenation of `[Wb[zrh], Rb[zrh]]` and "
            "`[WBb[zrh], RBb[zrh]]` (if bidirectional) along dimension 0. This tensor "
            "has shape `[num_directions, 6*hidden_size]`. Optional: If not specified "
            "- assumed to be 0",
            "T",
            OpSchema::Optional)
        .FillUsing(RNNDocGenerator2("GRU")));

static const char* LSTM_ver7_doc = R"DOC(
Computes an one-layer LSTM. This operator is usually supported via some
custom implementation such as CuDNN.

Notations:

`X` - input tensor

`i` - input gate

`o` - output gate

`f` - forget gate

`c` - cell gate

`t` - time step (t-1 means previous time step)

`W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates

`R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates

`Wb[iofc]` - W bias vectors for input, output, forget, and cell gates

`Rb[iofc]` - R bias vectors for input, output, forget, and cell gates

`P[iof]`  - P peephole weight vector for input, output, and forget gates

`WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates

`RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates

`WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates

`RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates

`PB[iof]`  - P peephole weight vector for backward input, output, and forget gates

`H` - Hidden state

`num_directions` - 2 if direction == bidirectional else 1

Activation functions:

  Relu(x)                - max(0, x)

  Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})

  Sigmoid(x)             - 1/(1 + e^{-x})

  (NOTE: Below are optional)

  Affine(x)              - alpha*x + beta

  LeakyRelu(x)           - x if x >= 0 else alpha * x

  ThresholdedRelu(x)     - x if x >= alpha else 0

  ScaledTanh(x)          - alpha*Tanh(beta*x)

  HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)

  Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)

  Softsign(x)            - x/(1 + |x|)

  Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

  - it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)

  - ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)

  - ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)

  - Ct = ft (.) Ct-1 + it (.) ct

  - ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)

  - Ht = ot (.) h(Ct)
)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    LSTM,
    7,
    OpSchema()
        .SetDoc(LSTM_ver7_doc + GenerateOptionalArgumentsDoc())
        .Attr(
            "activations",
            "A list of 3 (or 6 if bidirectional) activation functions "
            "for input, output, forget, cell, and hidden. The activation functions must "
            "be one of the activation functions specified above. Optional: See the equations "
            "for default if not specified.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr("input_forget", "Couple the input and forget gates if 1.", AttributeProto::INT, static_cast<int64_t>(0))
        .Input(
            1,
            "W",
            "The weight tensor for the gates. Concatenation of `W[iofc]` and "
            "`WB[iofc]` (if bidirectional) along dimension 0. The tensor has shape "
            "`[num_directions, 4*hidden_size, input_size]`.",
            "T")
        .Input(
            2,
            "R",
            "The recurrence weight tensor. Concatenation of `R[iofc]` and "
            "`RB[iofc]` (if bidirectional) along dimension 0. This tensor has shape "
            "`[num_directions, 4*hidden_size, hidden_size]`.",
            "T")
        .Input(
            3,
            "B",
            "The bias tensor for input gate. Concatenation of `[Wb[iofc], Rb[iofc]]`, "
            "and `[WBb[iofc], RBb[iofc]]` (if bidirectional) along dimension 0. This "
            "tensor has shape `[num_directions, 8*hidden_size]`. Optional: If not "
            "specified - assumed to be 0.",
            "T",
            OpSchema::Optional)
        .Input(
            6,
            "initial_c",
            "Optional initial value of the cell. If not specified - assumed "
            "to be 0. It has shape `[num_directions, batch_size, hidden_size]`.",
            "T",
            OpSchema::Optional)
        .Input(
            7,
            "P",
            "The weight tensor for peepholes. Concatenation of `P[iof]` and "
            "`PB[iof]` (if bidirectional) along dimension 0. It has shape "
            "`[num_directions, 3*hidde_size]`. Optional: If not specified - "
            "assumed to be 0.",
            "T",
            OpSchema::Optional)
        .FillUsing(RNNDocGenerator2("LSTM"))
        .Output(
            2,
            "Y_c",
            "The last output value of the cell. It has shape "
            "`[num_directions, batch_size, hidden_size]`.",
            "T",
            OpSchema::Optional));
} // namespace ONNX_NAMESPACE
