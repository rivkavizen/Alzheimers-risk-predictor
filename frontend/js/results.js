export function renderShapChart(canvas, shapExplanation) {
  const all = shapExplanation.all_shap || {};
  const entries = Object.entries(all)
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, 12);

  const labels = entries.map(([k]) => k);
  const values = entries.map(([, v]) => v);
  const colors = values.map((v) => (v >= 0 ? "rgba(220,38,38,0.7)" : "rgba(22,163,74,0.7)"));

  return new Chart(canvas, {
    type: "bar",
    data: {
      labels,
      datasets: [{ label: "SHAP value", data: values, backgroundColor: colors }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { title: { display: true, text: "Impact on risk" } } },
    },
  });
}

export function renderRecommendations(container, recommendations) {
  if (!recommendations?.length) {
    container.innerHTML = "<p class='muted'>No recommendations available. Add your API key in Settings.</p>";
    return;
  }
  container.innerHTML = recommendations.map((r) => `
    <div class="recommendation priority-${r.priority || "medium"}">
      <strong>${r.category || "General"}</strong> (${r.priority || "medium"})
      <p>${r.action || ""}</p>
      <p class="muted"><em>${r.evidence || ""}</em></p>
    </div>
  `).join("");
}
