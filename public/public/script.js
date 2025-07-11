document.getElementById("predictBtn").addEventListener("click", async () => {
  const hr = document.getElementById("hr").value;
  const temp = document.getElementById("temp").value;
  const act = document.getElementById("act").value;

  const res = await fetch(`/predict?hr=${hr}&temp=${temp}&act=${act}`);
  const data = await res.json();

  if (data.error) {
    document.getElementById("result").textContent = data.error;
  } else {
    document.getElementById("result").textContent =
      `스트레스 확률: ${(data.probability * 100).toFixed(2)}% → ` +
      (data.stress ? "🔴 스트레스 상태" : "🟢 정상 상태");
  }
});
