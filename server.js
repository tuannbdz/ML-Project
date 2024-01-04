const express = require("express");
const axios = require("axios");
const path = require("path");

const app = express();
const port = 3000;

app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use("/public", express.static(`${__dirname}/public`));

// Route to handle the POST request
app.post("/api/model", async (req, res) => {
  // Send a POST request to the Python server
  try {
    const response = await axios.post(
      "http://127.0.0.1:5000/api/model",
      req.body
    );
    res.json(response.data);
  } catch (error) {
    res.status(error.response ? error.response.status : 500).json({
      error: error.response ? error.response.data : error.message,
    });
  }
});

app.all("*", (req, res) => {
  res.sendFile(path.join(__dirname, "/public/index.html"));
});

app.listen(port, () => {
  console.log(`App running on http://127.0.0.1:${port}`);
});
