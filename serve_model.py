from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Model directory and tags
model_dir = "car_ids_model"
tags = [tf.compat.v1.saved_model.tag_constants.SERVING]

# Load the model into a session
sess = tf.compat.v1.Session(graph=tf.Graph())
tf.compat.v1.saved_model.loader.load(sess, tags, model_dir)

# Define the prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse input data from the request
        data = request.get_json(force=True)
        
        # Convert input data to the appropriate format
        input_data = np.array(data['input'], dtype=np.float32)
        input_data = input_data.flatten().reshape(1,841)

        # Locate the input and output tensors
        input_tensor = sess.graph.get_tensor_by_name('Labeled_Input:0')   # replace 'input_tensor_name' with your model's input tensor name
        keep_prob_tensor = sess.graph.get_tensor_by_name('keep_prob:0')   # replace 'input_tensor_name' with your model's input tensor name
        output_tensor = sess.graph.get_tensor_by_name('ArgMax:0')  # replace 'output_tensor_name' with your model's output tensor name
        
        # Run the session to get predictions
        predictions = sess.run(output_tensor, feed_dict={input_tensor: input_data, keep_prob_tensor: 1.0})

        # Return the predictions as JSON
        return jsonify({"predictions": predictions.tolist()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route("/health", methods=["GET"])
def health():
    try:
        # Check if the session and graph are available
        if sess is not None and sess.graph is not None:
            return jsonify({"status": "UP"}), 200
        else:
            return jsonify({"status": "DOWN"}), 500
    except Exception as e:
        return jsonify({"status": "DOWN", "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
