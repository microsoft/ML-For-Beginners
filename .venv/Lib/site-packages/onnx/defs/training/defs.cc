/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cmath>

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

static const char* Gradient_ver1_doc = R"DOC(
Gradient operator computes the partial derivatives of a specific tensor w.r.t.
some other tensors. This operator is widely used in gradient-based training
algorithms. To illustrate its use, let's consider a computation graph,

```
X -----.
       |
       v
W --> Conv --> H --> Gemm --> Y
                      ^
                      |
                      Z
```

, where W and Z are trainable tensors. Note that operators' attributes are
omitted for the sake of simplicity. Let dY/dW (dY/dZ) be the gradient of
Y with respect to W (Z). The user can compute gradient by inserting Gradient
operator to form another graph shown below.

```
W --> Conv --> H --> Gemm --> Y
|      ^              ^
|      |              |
|      X              Z
|      |              |
|      |   .----------'
|      |   |  (W/Z/X is the 1st/2nd/3rd input of Gradient as shown in
|      |   |   "xs" followed by "zs")
|      v   v
'---> Gradient(xs=["W", "Z"], zs=["X"], y="Y")
       |   |
       |   '-----------------------------------> dY/dW (1st output of Gradient)
       |
       '---------------------------------------> dY/dZ (2nd output of Gradient)
```

By definition, the tensor "y" is a function of independent variables in "xs"
and "zs". Since we only compute the gradient of "y" w.r.t. the differentiable
variables in "xs", this Gradient only outputs dY/dW and dY/dZ. Note that "H"
cannot appear in "xs" and "zs". The reason is that "H" can be determined by
tensors "W" and "X" and therefore "H" is not an independent variable.

All outputs are optional. If needed, for example, user can assign an empty
string to the 1st output name of that Gradient to skip the generation of dY/dW.
Note that the concept of optional outputs can also be found in ONNX's RNN, GRU,
and LSTM.

Gradient operator can compute derivative against intermediate tensors. For
example, the gradient of Y with respect to H can be done via

```
W --> Conv --> H --> Gemm --> Y
       ^       |      ^
       |       |      |
       X       |      Z
       .-------'      |
       |   .----------'
       |   | (H/Z is the 1st/2nd input of Gradient as shown in "xs")
       v   v
      Gradient(xs=["H", "Z"], y="Y")
       |   |
       |   '-----------------------------------> dY/dH (1st output of Gradient)
       |
       '---------------------------------------> dY/dZ (2nd output of Gradient)
```

It is possible to represent high-order differentiation using Gradient operators.
For example, given the following linear model:

```
W --> Gemm --> Y --> Loss --> O
       ^              ^
       |              |
       X              L
```

To compute the 2nd order derivative of O with respect to W (denoted by
d^2O/dW^2), one can do

```
W --> Gemm --> Y --> Loss --> O
|      ^              ^
|      |              |
|      X .------------L
|      | |            |
|      | |            v
+------+-+> Gradient(xs=["X", "W"], zs=["L"], y="O") ---> dO/dX (1st output of Gradient)
|      | |    |
|      | |    '---> dO/dW (2nd output of Gradient)
|      v v
'---> Gradient(xs=["X", "W"], zs=["L"], y="dO/dW") ---> d(dO/dW)dX (1st output of
       |                                                  Gradient)
       |
       |
       '---> d^2O/dW^2 (2nd output of Gradient)
```

The tensors named in attributes "xs", "zs", and "y" define the differentiated
computation graph, and the inputs to Gradient node define the values at
which the gradient is computed. We can feed different tensors to the identified
graph. For example, one can compute the gradient of Y with respect to H at
a specific value of H, H_1, by providing that value as an input to the Gradient
node.

```
W --> Conv --> H --> Gemm --> Y
       ^              ^
       |              |
       X              Z

          Z_1 (2nd input of Gradient)
           |
           v
H_1 --> Gradient(xs=["H", "Z"], y="Y") ---> dY/dH when H = H_1 and Y = Y_1.
           |
           '------------------------------> dY/dZ (2nd output of Gradient)
```

When the inputs of Gradient are the tensors named in "xs" and "zs", the
computation can be optimized. More specifically, intermediate variables in
forward pass can be reused if the gradient is computed via reverse-mode
auto-differentiation.

)DOC";

ONNX_PREVIEW_TRAINING_OPERATOR_SET_SCHEMA(
    Gradient,
    1,
    OpSchema()
        .SetDoc(Gradient_ver1_doc)
        .Input(
            0,
            "Inputs",
            "The values fed into graph identified by the attributes. "
            "The i-th input is the value of the i-th tensor specified in the "
            "concatenated list of the attribute \"xs\" and the attribute "
            " \"zs\". For example, if xs=[\"A\", \"B\"] and zs=[\"C\"], the "
            "first input is used as the value of symbol \"A\" and the 3rd "
            "input is substituted for all the occurrences of \"C\".",
            "T1",
            OpSchema::Variadic,
            false)
        .Output(
            0,
            "Outputs",
            "The gradient of the tensor specified by the attribute \"y\" "
            "with respect to each of tensors specified in the "
            "attribute \"xs\". The i-th output is the gradient of \"y\" with "
            "respect to the i-th tensor specified in the attribute \"xs\".",
            "T2",
            OpSchema::Variadic,
            false)
        .Attr(
            "xs",
            "Input tensor names of the differentiated sub-graph. It "
            "contains only the necessary differentiated "
            "inputs of a (sub-)graph. Variables (usually called "
            "intermediate variables) that can be generated from inputs "
            "cannot be included in this attribute.",
            AttributeProto::STRINGS)
        .Attr(
            "zs",
            "Input tensor names of the differentiated sub-graph. It "
            "contains only the necessary non-differentiated "
            "inputs of a (sub-)graph. Variables (usually called "
            "intermediate variables) that can be generated from inputs "
            "cannot be included in this attribute.",
            AttributeProto::STRINGS,
            OPTIONAL_VALUE)
        .Attr(
            "y",
            "The targeted tensor. It can be viewed as the output of the "
            "differentiated function. The attribute \"xs\" and attribute "
            "\"zs\" are the minimal independent variable set that determines "
            "the value of \"y\".",
            AttributeProto::STRING)
        .TypeConstraint("T1", OpSchema::all_tensor_types(), "Allow outputs to be any kind of tensor.")
        .TypeConstraint(
            "T2",
            {"tensor(float16)", "tensor(float)", "tensor(double)"},
            "Allow inputs to be any kind of floating-point tensor."));

static const char* Adagrad_ver1_doc = R"DOC(
    Compute one iteration of ADAGRAD, a stochastic gradient based optimization
    algorithm. This operator can conduct the optimization of multiple tensor variables.

    Let's define the behavior of this operator. As you can imagine, ADAGRAD requires
    some parameters:

     - The initial learning-rate "R".
     - The update count "T". That is, the number of training iterations conducted.
     - A L2-norm regularization coefficient "norm_coefficient".
     - A learning-rate decay factor "decay_factor".
     - A small constant "epsilon" to avoid dividing-by-zero.

    At each ADAGRAD iteration, the optimized tensors are moved along a direction
    computed based on their estimated gradient and accumulated squared gradient. Assume
    that only a single tensor "X" is updated by this operator. We need the value of "X",
    its gradient "G", and its accumulated squared gradient "H". Therefore, variables in
    this operator's input list are sequentially "R", "T", "X", "G", and "H". Other
    parameters are given as attributes because they are usually constants. Also, the
    corresponding output tensors are the new value of "X" (called "X_new"), and then
    the new accumulated squared gradient (called "H_new"). Those outputs are computed
    from the given inputs following the pseudo code below.

    Let "+", "-", "*", and "/" are all element-wise arithmetic operations with
    numpy-style broadcasting support. The pseudo code to compute those outputs is:

      // Compute a scalar learning-rate factor. At the first update of X, T is generally
      // 0 (0-based update index) or 1 (1-based update index).
      r = R / (1 + T * decay_factor);

      // Add gradient of 0.5 * norm_coefficient * ||X||_2^2, where ||X||_2 is the 2-norm.
      G_regularized = norm_coefficient * X + G;

      // Compute new accumulated squared gradient.
      H_new = H + G_regularized * G_regularized;

      // Compute the adaptive part of per-coordinate learning rate. Note that Sqrt(...)
      // computes element-wise square-root.
      H_adaptive = Sqrt(H_new) + epsilon

      // Compute the new value of "X".
      X_new = X - r * G_regularized / H_adaptive;

    If one assign this operators to optimize multiple inputs, for example, "X_1" and "X_2", the same
    pseudo code may be extended to handle all tensors jointly. More specifically, we can view "X" as a
    concatenation of "X_1" and "X_2" (of course, their gradient and accumulate gradient should
    be concatenated too) and then just reuse the entire pseudo code.

    Note that ADAGRAD was first proposed in http://jmlr.org/papers/volume12/duchi11a/duchi11a.pdf.
    In that reference paper, this operator is a special case of the Figure 1's composite mirror
    descent update.
)DOC";

ONNX_PREVIEW_TRAINING_OPERATOR_SET_SCHEMA(
    Adagrad,
    1,
    OpSchema()
        .SetDoc(Adagrad_ver1_doc)
        .Input(0, "R", "The initial learning rate.", "T1")
        .Input(1, "T", "The update count of \"X\". It should be a scalar.", "T2")
        .Input(
            2,
            "inputs",
            "The current values of optimized tensors, followed by their "
            "respective gradients, followed by their respective accumulated squared gradients."
            "For example, if two tensor \"X_1\" and \"X_2\" "
            "are optimized, "
            "The input list would be "
            "[\"X_1\", \"X_2\", "
            "gradient of \"X_1\", "
            "gradient of \"X_2\", "
            "accumulated squared gradient of \"X_1\", "
            "accumulated squared gradient of \"X_2\"].",
            "T3",
            OpSchema::Variadic,
            false)
        .Output(
            0,
            "outputs",
            "Updated values of optimized tensors, followed by their updated "
            "values of accumulated squared gradients. For example, "
            "if two tensor \"X_1\" and \"X_2\" are "
            "optimized, the output list would be [new value of \"X_1,\" new value of \"X_2\" "
            "new accumulated squared gradient of \"X_1\", new accumulated squared gradient of \"X_2\"].",
            "T3",
            OpSchema::Variadic,
            false)
        .Attr("epsilon", "Small scalar to avoid dividing by zero.", AttributeProto::FLOAT, 1e-6f)
        .Attr(
            "decay_factor",
            "The decay factor of learning rate after one update."
            "The effective learning rate is computed by r = R / (1 + T * decay_factor). "
            "Default to 0 so that increasing update counts doesn't reduce the learning rate.",
            AttributeProto::FLOAT,
            0.0f)
        .Attr(
            "norm_coefficient",
            "Regularization coefficient in 0.5 * norm_coefficient * ||X||_2^2. Default to 0, "
            "which means no regularization.",
            AttributeProto::FLOAT,
            0.0f)
        .TypeConstraint("T1", {"tensor(float)", "tensor(double)"}, "Constrain input types to float scalars.")
        .TypeConstraint("T2", {"tensor(int64)"}, "Constrain input types to 64-bit integer scalars.")
        .TypeConstraint("T3", {"tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // In comments below, we assume that the input list is
          // [R, T, X1, X2, G1, G2, H1, H2] and the output list is
          // [X1_new, X2_new, H1_new, H2_new].

          // Compute the number of tuples (X, G, H).
          auto num_optimized_tensors = (ctx.getNumInputs() - 2) / 3;
          for (size_t i = 0; i < num_optimized_tensors; ++i) {
            // Pass X1's and X2's shapes to X1_new and X2_new, respectively.
            size_t i_in = 2 + i;
            size_t i_out = i;
            propagateElemTypeFromInputToOutput(ctx, i_in, i_out);
            propagateShapeFromInputToOutput(ctx, i_in, i_out);

            // Pass H1's and H2's shapes to H1_new and H2_new, respectively.
            i_in = 2 + 2 * num_optimized_tensors + i;
            i_out = i + num_optimized_tensors;
            propagateElemTypeFromInputToOutput(ctx, i_in, i_out);
            propagateShapeFromInputToOutput(ctx, i_in, i_out);
          }
        }));

static const char* Momentum_ver1_doc = R"DOC(
    Compute one iteration of stochastic gradient update with momentum.
    This operator can conduct the optimization of multiple tensor variables.

    Let's define the behavior of this operator. As you can imagine, SG with momentum requires
    several parameters:

     - The learning-rate "R".
     - The update count "T". That is, the number of conducted training iterations. It should
       be zero in the first training iteration.
     - A L2-norm regularization coefficient "norm_coefficient".
     - A decay coefficient of previous accumulated gradient (i.e., momentum) "alpha".
     - The scaling coefficient of current gradient "beta".
     - An attribute to choose either standard momentum or Nesterov's momentum "mode" should
       be used.

    For the sake of simplicity, assume that there is only one tensor (called "X") to be optimized.
    Other necessary inputs are "X"'s gradient (called "G") and "X"'s momentum (called "V"). This
    Momentum operator maps all these inputs to the new value of "X" (called "X_new") and its new
    momentum (called "V_new").

    This operator supports two different momentum algorithms. Set the attribute "mode" to
    "nesterov" if Nesterov's momentum is desired. Otherwise, set the attribute "model" to
    "standard" to use standard momentum. Computation details are described subsequently.

    Let "+", "-", "*", and "/" are all element-wise operations with numpy-style broadcasting.

    Pseudo code for SG with standard momentum:

      // Add gradient of 0.5 * norm_coefficient * ||X||^2, where ||X|| is the sum of squared
      // values of all elements in X.
      G_regularized = norm_coefficient * X + G

      // In the first training iteration, beta should always be 1.
      beta_adjusted = T > 0 ? beta : 1

      // Compute the current momentum based on previous momentum and the current gradient.
      V_new = alpha * V + beta_adjusted * G_regularized

      // Update X.
      X_new = X - R * V_new

    Pseudo code for SG with Nesterov's momentum:

      // Add gradient of 0.5 * norm_coefficient * ||X||^2, where ||X|| is the sum of squared
      // values of all elements in X.
      G_regularized = norm_coefficient * X + G;

      // In the first training iteration, beta should always be 1.
      beta_adjusted = T > 0 ? beta : 1

      // Compute the current momentum based on previous momentum and the current gradient.
      V_new = alpha * V + beta_adjusted * G_regularized;

      // Compute final update direction and then update X.
      X_new = X - R * (G_regularized + alpha * V_new)

    If one assign this operators to optimize multiple inputs, for example, "X_1" and "X_2". The same
    pseudo code would be extended to handle all tensors jointly. More specifically, we can view "X" as a
    concatenation of "X_1" and "X_2" (of course, their gradient and accumulate gradient should
    be concatenated too) and then our pseudo code becomes applicable.
)DOC";

ONNX_PREVIEW_TRAINING_OPERATOR_SET_SCHEMA(
    Momentum,
    1,
    OpSchema()
        .SetDoc(Momentum_ver1_doc)
        .Input(0, "R", "The learning rate.", "T1")
        .Input(1, "T", "Update count of \"X\". It should be a scalar.", "T2")
        .Input(
            2,
            "inputs",
            "It sequentially contains the current values of optimized tensors, then their "
            "gradient tensors, and finally their momentum tensors. For example, if two tensors "
            "\"X_1\" and \"X_2\" are optimized, The expected input list would be "
            "[\"X_1\", \"X_2\", gradient of \"X_1\", gradient of \"X_2\", momentum of \"X_1\", momentum of \"X_2\"].",
            "T3",
            OpSchema::Variadic,
            false)
        .Output(
            0,
            "outputs",
            "It sequentially contains the new values of optimized tensors and then the new "
            "values of their momentum tensors. For example, if two tensors \"X_1\" and \"X_2\" are "
            "optimized, the output list would be [new value of \"X_1,\" new value of \"X_2\" "
            "new momentum of \"X_1\", new momentum of \"X_2\"].",
            "T3",
            OpSchema::Variadic,
            false)
        .Attr("alpha", "The decay factor of momentum. It should be a scalar.", AttributeProto::FLOAT)
        .Attr(
            "beta",
            "The coefficient of gradient in computing new momentum. It should be a scalar.",
            AttributeProto::FLOAT)
        .Attr("norm_coefficient", "Coefficient of 0.5 * norm_coefficient * ||X||^2.", AttributeProto::FLOAT)
        .Attr(
            "mode",
            "Its value should be either \"nesterov\" or \"standard\". The value \"nesterov\" leads "
            "to the use of Nesterov's momentum while \"standard\" invokes stochastic gradient method "
            "using standard momentum",
            AttributeProto::STRING)
        .TypeConstraint("T1", {"tensor(float)", "tensor(double)"}, "Constrain input types to float scalars.")
        .TypeConstraint("T2", {"tensor(int64)"}, "Constrain input types to 64-bit integer scalars.")
        .TypeConstraint("T3", {"tensor(float)", "tensor(double)"}, "Constrain input types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Assume that the input list is [R, T, X1, X2, G1, G2, V1, V2] and
          // output list is [X1_new, X2_new, V1_new, V2_new] for explaining
          // the code below in a simpler way.

          // The count of input tensors excluding "R" and "T".
          auto num_adjustable_tensors = ctx.getNumInputs() - 2;

          // Check number of (optimized tensor, gradient, momentum) tuples.
          if (num_adjustable_tensors % 3 != 0) {
            fail_shape_inference(
                "The sum of optimized tensor count and momentum tensor count ",
                "should be a multiple of 2 in the input list of Momentum operator");
          }

          // The count of "X1" and "X2".
          auto num_optimized_tensors = num_adjustable_tensors / 3;
          for (size_t i = 0; i < num_optimized_tensors; ++i) {
            // Pass X1's/X2's shapes to X1_new/X2_new.
            size_t i_in = 2 + i;
            size_t i_out = i;
            propagateElemTypeFromInputToOutput(ctx, i_in, i_out);
            propagateShapeFromInputToOutput(ctx, i_in, i_out);
            // Pass V1's/V2's shapes to V1_new/V2_new.
            i_in = 2 + 2 * num_optimized_tensors + i;
            i_out = i + num_optimized_tensors;
            propagateElemTypeFromInputToOutput(ctx, i_in, i_out);
            propagateShapeFromInputToOutput(ctx, i_in, i_out);
          }
        }));

static const char* Adam_ver1_doc = R"DOC(
    Compute one iteration of Adam, a stochastic gradient based optimization
    algorithm. This operator can conduct the optimization of multiple tensor variables.

    Let's define the behavior of this operator. First of all, Adam requires
    some parameters:

     - The learning-rate "R".
     - The update count "T". That is, the number of training iterations conducted.
     - A L2-norm regularization coefficient "norm_coefficient".
     - A small constant "epsilon" to avoid dividing-by-zero.
     - Two coefficients, "alpha" and "beta".

    At each Adam iteration, the optimized tensors are moved along a direction
    computed based on their exponentially-averaged historical gradient and
    exponentially-averaged historical squared gradient. Assume that only a tensor
    "X" is being optimized. The rest of required information is

     - the value of "X",
     - "X"'s gradient (denoted by "G"),
     - "X"'s exponentially-averaged historical gradient (denoted by "V"), and
     - "X"'s exponentially-averaged historical squared gradient (denoted by "H").

    Some of those parameters are passed into this operator as input tensors and others
    are stored as this operator's attributes. Specifically, this operator's input tensor
    list is ["R", "T", "X", "G", "V", "H"]. That is, "R" is the first input, "T" is
    the second input, and so on. Other parameters are given as attributes because they
    are constants. Moreover, the corresponding output tensors are

     - the new value of "X" (called "X_new"),
     - the new exponentially-averaged historical gradient (denoted by "V_new"), and
     - the new exponentially-averaged historical squared gradient (denoted by "H_new").

    Those outputs are computed following the pseudo code below.

    Let "+", "-", "*", and "/" are all element-wise arithmetic operations with
    numpy-style broadcasting support. The pseudo code to compute those outputs is:

      // Add gradient of 0.5 * norm_coefficient * ||X||_2^2, where ||X||_2 is the 2-norm.
      G_regularized = norm_coefficient * X + G

      // Update exponentially-averaged historical gradient.
      V_new = alpha * V + (1 - alpha) * G_regularized

      // Update exponentially-averaged historical squared gradient.
      H_new = beta * H + (1 - beta) * G_regularized * G_regularized

      // Compute the element-wise square-root of H_new. V_new will be element-wisely
      // divided by H_sqrt for a better update direction.
      H_sqrt = Sqrt(H_new) + epsilon

      // Compute learning-rate. Note that "alpha**T"/"beta**T" is alpha's/beta's T-th power.
      R_adjusted = T > 0 ? R * Sqrt(1 - beta**T) / (1 - alpha**T) : R

      // Compute new value of "X".
      X_new = X - R_adjusted * V_new / H_sqrt

      // Post-update regularization.
      X_final = (1 - norm_coefficient_post) * X_new

    If there are multiple inputs to be optimized, the pseudo code will be applied
    independently to each of them.
)DOC";

ONNX_PREVIEW_TRAINING_OPERATOR_SET_SCHEMA(
    Adam,
    1,
    OpSchema()
        .SetDoc(Adam_ver1_doc)
        .Input(0, "R", "The initial learning rate.", "T1")
        .Input(1, "T", "The update count of \"X\". It should be a scalar.", "T2")
        .Input(
            2,
            "inputs",
            "The tensors to be optimized, followed by their respective gradients, "
            "followed by their respective accumulated gradients (aka momentum), "
            "followed by their respective accumulated squared gradients. For example, "
            "to optimize tensors \"X_1\" and \"X_2,\", the input list would be "
            "[\"X_1\", \"X_2\", "
            "gradient of \"X_1\", gradient of \"X_2\", "
            "accumulated gradient of \"X_1\", accumulated gradient of \"X_2\", "
            "accumulated squared gradient of \"X_1\", accumulated squared gradient of \"X_2\"].",
            "T3",
            OpSchema::Variadic,
            false)
        .Output(
            0,
            "outputs",
            "New values of optimized tensors, "
            "followed by their respective new accumulated gradients, "
            "followed by their respective new accumulated squared gradients. "
            "For example, if two tensors \"X_1\" and \"X_2\" are optimized, "
            "the outputs list would be "
            "[new value of \"X_1\", new value of \"X_2\", "
            "new accumulated gradient of \"X_1\", "
            "new accumulated gradient of \"X_2\", "
            "new accumulated squared gradient of \"X_1\", "
            "new accumulated squared gradient of \"X_2\"].",
            "T3",
            OpSchema::Variadic,
            false)
        .Attr(
            "alpha",
            "Coefficient of previously accumulated gradient in running average. Default to 0.9.",
            AttributeProto::FLOAT,
            0.9f)
        .Attr(
            "beta",
            "Coefficient of previously accumulated squared-gradient in running average. Default to 0.999.",
            AttributeProto::FLOAT,
            0.999f)
        .Attr(
            "norm_coefficient",
            "Regularization coefficient of 0.5 * norm_coefficient * ||X||_2^2. Default to 0, "
            "which means no regularization.",
            AttributeProto::FLOAT,
            0.0f)
        .Attr(
            "norm_coefficient_post",
            "Regularization coefficient of 0.5 * norm_coefficient * ||X||_2^2. Default to 0, "
            "which means no regularization.",
            AttributeProto::FLOAT,
            0.0f)
        .Attr("epsilon", "Small scalar to avoid dividing by zero.", AttributeProto::FLOAT, 1e-6f)
        .TypeConstraint("T1", {"tensor(float)", "tensor(double)"}, "Constrain input types to float scalars.")
        .TypeConstraint("T2", {"tensor(int64)"}, "Constrain input types to 64-bit integer scalars.")
        .TypeConstraint("T3", {"tensor(float)", "tensor(double)"}, "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Assume that the input list is [R, T, X1, X2, G1, G2, V1, V2, H1, H2] and
          // output list is [X1_new, X2_new, V1_new, V2_new, H1_new, H2_new] for explaining
          // the code below in a simpler way.

          // The count of input tensors excluding "R" and "T".
          auto num_adjustable_tensors = ctx.getNumInputs() - 2;

          // Check number of (optimized tensor, gradient, momentum) tuples.
          if (num_adjustable_tensors % 4 != 0) {
            fail_shape_inference(
                "The sum of optimized tensor count, gradient tensor count, momentum tensor count, ",
                "accumulated squared-gradient tensor count should be a multiple of 4 in the ",
                "\"inputs\" of Adam operator.");
          }

          // The count of "X1" and "X2".
          auto num_optimized_tensors = num_adjustable_tensors / 4;
          for (size_t i = 0; i < num_optimized_tensors; ++i) {
            // Pass X1's/X2's shapes to X1_new/X2_new.
            size_t i_in = 2 + i;
            size_t i_out = i;
            propagateElemTypeFromInputToOutput(ctx, i_in, i_out);
            propagateShapeFromInputToOutput(ctx, i_in, i_out);

            // Pass V1's/V2's shapes to V1_new/V2_new.
            i_in = 2 + 2 * num_optimized_tensors + i;
            i_out = num_optimized_tensors + i;
            propagateElemTypeFromInputToOutput(ctx, i_in, i_out);
            propagateShapeFromInputToOutput(ctx, i_in, i_out);

            // Pass H1's/H2's shapes to H1_new/H2_new.
            i_in = 2 + 3 * num_optimized_tensors + i;
            i_out = 2 * num_optimized_tensors + i;
            propagateElemTypeFromInputToOutput(ctx, i_in, i_out);
            propagateShapeFromInputToOutput(ctx, i_in, i_out);
          }
        }));

} // namespace ONNX_NAMESPACE
