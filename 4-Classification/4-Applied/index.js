const ingredients = Array(380).fill(0);

const checks = [...document.querySelectorAll(".checkbox")];

checks.forEach((check) => {
  check.addEventListener("change", function () {
    // toggle the state of the ingredient
    // based on the checkbox's value (1 or 0)
    ingredients[check.value] = check.checked ? 1 : 0;
  });
});

function testCheckboxes() {
  // validate if at least one checkbox is checked
  return checks.some((check) => check.checked);
}

async function startInference() {
  let atLeastOneChecked = testCheckboxes();

  if (!atLeastOneChecked) {
    alert("Please select at least one ingredient.");
    return;
  }
  try {
    // create a new session and load the model.

    const session = await ort.InferenceSession.create("./model.onnx");

    const input = new ort.Tensor(new Float32Array(ingredients), [1, 380]);
    const feeds = { float_input: input };

    // feed inputs and run
    const results = await session.run(feeds);

    // read from results
    alert("You can enjoy " + results.label.data[0] + " cuisine today!");
  } catch (e) {
    console.log(`failed to inference ONNX model`);
    console.error(e);
  }
}
