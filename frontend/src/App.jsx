import React, { useState } from "react";
import "./App.css";

function App() {
  const [formData, setFormData] = useState({
    cost_deviation: 2500,
    worker_count: 15,
    equipment_utilization_rate: 65,
    material_usage: 500,
    vendor_remarks:
      "Severe cement shortage from the primary supplier, holding up the foundation pouring.",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const analyzeRisk = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cost_deviation: parseFloat(formData.cost_deviation),
          worker_count: parseInt(formData.worker_count),
          equipment_utilization_rate: parseFloat(
            formData.equipment_utilization_rate,
          ),
          material_usage: parseFloat(formData.material_usage),
          vendor_remarks: formData.vendor_remarks,
        }),
      });

      const data = await response.json();
      setResult(data);
    } catch (error) {
      alert(
        "Error connecting to the Grid-Guard API. Is the Python server running?",
      );
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard">
      <h1>Grid-Guard Risk Radar</h1>

      <form onSubmit={analyzeRisk}>
        <div className="form-group">
          <label>Cost Deviation (USD):</label>
          <input
            type="number"
            name="cost_deviation"
            value={formData.cost_deviation}
            onChange={handleChange}
            required
          />
        </div>

        <div className="form-group">
          <label>Current Worker Count:</label>
          <input
            type="number"
            name="worker_count"
            value={formData.worker_count}
            onChange={handleChange}
            required
          />
        </div>

        <div className="form-group">
          <label>Equipment Utilization (%):</label>
          <input
            type="number"
            name="equipment_utilization_rate"
            value={formData.equipment_utilization_rate}
            onChange={handleChange}
            required
          />
        </div>

        <div className="form-group">
          <label>Material Usage (kg):</label>
          <input
            type="number"
            name="material_usage"
            value={formData.material_usage}
            onChange={handleChange}
            required
          />
        </div>

        <div className="form-group">
          <label>Latest Vendor/Engineer Log:</label>
          <textarea
            name="vendor_remarks"
            rows="3"
            value={formData.vendor_remarks}
            onChange={handleChange}
            required
          />
        </div>

        <button type="submit" disabled={loading}>
          {loading ? "Analyzing..." : "Analyze Project Risk"}
        </button>
      </form>

      {result && (
        <div
          className={`result-box ${result.raw_status === 1 ? "risk" : "safe"}`}
        >
          Prediction: <span>{result.prediction}</span>
        </div>
      )}
    </div>
  );
}

export default App;
