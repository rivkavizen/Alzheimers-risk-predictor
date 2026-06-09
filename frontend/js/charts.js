/** Y-axis 0–1 so risk scores are easy to interpret (0% = low risk, 100% = high). */
export function riskChartOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        min: 0,
        max: 1,
        ticks: {
          stepSize: 0.2,
          callback: (v) => `${(v * 100).toFixed(0)}%`,
        },
        title: { display: true, text: "Risk probability" },
      },
    },
    plugins: {
      tooltip: {
        callbacks: {
          label: (ctx) => `Risk: ${(ctx.parsed.y * 100).toFixed(1)}%`,
        },
      },
    },
  };
}

/**
 * Map cognitive metrics to 0–1 using clinical max scales.
 * ADL is inverted (lower raw ADL = better independence).
 */
export function buildCognitiveSeries(assessments) {
  const mmse = [];
  const functional = [];
  const adlInverted = [];
  const raw = [];

  for (const a of assessments) {
    const mmseVal = Number(a.mmse);
    const faVal = Number(a.functional_assessment);
    const adlVal = Number(a.adl);
    mmse.push(mmseVal / 30);
    functional.push(faVal / 10);
    adlInverted.push(1 - adlVal / 10);
    raw.push({ mmse: mmseVal, functional_assessment: faVal, adl: adlVal });
  }

  return { mmse, functional, adlInverted, raw };
}

export function cognitiveChartOptions(rawRows) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      y: {
        min: 0,
        max: 1,
        ticks: {
          stepSize: 0.2,
          callback: (v) => `${(v * 100).toFixed(0)}%`,
        },
        title: { display: true, text: "% of best score (higher = better)" },
      },
    },
    plugins: {
      tooltip: {
        callbacks: {
          label: (ctx) => {
            const i = ctx.dataIndex;
            const r = rawRows[i];
            const label = ctx.dataset.label || "";
            if (label.includes("MMSE")) return `MMSE: ${r.mmse.toFixed(2)} / 30`;
            if (label.includes("Functional")) return `Functional: ${r.functional_assessment.toFixed(2)} / 10`;
            if (label.includes("ADL")) return `ADL: ${r.adl.toFixed(2)} / 10 (lower raw = better)`;
            return `${label}: ${(ctx.parsed.y * 100).toFixed(0)}%`;
          },
        },
      },
      legend: { position: "bottom" },
    },
    elements: {
      point: { radius: 5, hoverRadius: 7 },
      line: { tension: 0.2 },
    },
  };
}
