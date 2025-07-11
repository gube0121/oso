/********************************************************************
 * 1) 간단한 심박수·체온·활동량 → 스트레스 여부 모델을 tfjs로 학습
 * 2) /predict?hr=..&temp=..&act=..  REST API로 확률 반환
 ********************************************************************/
import express from "express";
import * as tf from "@tensorflow/tfjs-node";

const app = express();
const PORT = process.env.PORT || 3000;
app.use(express.static("public"));   //  public 폴더 정적파일 제공

/* ---------- 1. 합성 데이터 생성 ---------- */
function makeSample(label) {
  if (label === 1) { // 스트레스
    return [
      tf.randomNormal([1], 100, 10).dataSync()[0],   // HR
      tf.randomNormal([1], 37.5, 0.5).dataSync()[0], // Temp
      tf.randomNormal([1], 80, 20).dataSync()[0]     // Activity
    ];
  }
  // 정상
  return [
    tf.randomNormal([1], 75, 8).dataSync()[0],
    tf.randomNormal([1], 36.5, 0.3).dataSync()[0],
    tf.randomNormal([1], 50, 15).dataSync()[0]
  ];
}
const N = 1000;
const xs = [];
const ys = [];
for (let i = 0; i < N; i++) {
  const label = Math.random() < 0.3 ? 1 : 0;
  xs.push(makeSample(label));
  ys.push(label);
}
const xTensor = tf.tensor2d(xs);
const yTensor = tf.tensor2d(ys, [N, 1]);

/* ---------- 2. 모델 정의 & 학습 ---------- */
const model = tf.sequential();
model.add(tf.layers.dense({ units: 16, activation: "relu", inputShape: [3] }));
model.add(tf.layers.dense({ units: 8, activation: "relu" }));
model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
model.compile({ optimizer: "adam", loss: "binaryCrossentropy", metrics: ["accuracy"] });

console.log("⏳ Training the model…");
await model.fit(xTensor, yTensor, {
  epochs: 20,
  batchSize: 16,
  validationSplit: 0.2,
  verbose: 0
});
console.log("✅ Model trained.");

/* ---------- 3. 추론용 REST 엔드포인트 ---------- */
app.get("/predict", async (req, res) => {
  const { hr, temp, act } = req.query;                  // 쿼리 파라미터
  const nums = [Number(hr), Number(temp), Number(act)];
  if (nums.some((v) => isNaN(v))) {
    return res.status(400).json({ error: "Invalid numeric input." });
  }
  const input = tf.tensor2d([nums]);
  const prob = (await model.predict(input).data())[0];  // 확률
  res.json({ probability: prob, stress: prob > 0.5 });
});

/* ---------- 4. 서버 실행 ---------- */
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
