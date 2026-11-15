// Interactive Logic for GitHub Pages ML Showcase

let currentTask = "fraud"; // 'fraud' or 'loan'
let simData = null;
let chartInstance = null;

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", async () => {
  initTheme();
  setupEventListeners();
  await loadSimulatorData();
  updateSimulatorView();
});

// Theme Management
function initTheme() {
  const savedTheme = localStorage.getItem("finrisk-theme") || "dark";
  document.documentElement.setAttribute("data-theme", savedTheme);
  updateThemeButton(savedTheme);

  document.getElementById("themeToggleBtn").addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("finrisk-theme", next);
    updateThemeButton(next);
    if (chartInstance) renderChart();
  });
}

function updateThemeButton(theme) {
  const btn = document.getElementById("themeToggleBtn");
  if (btn) {
    btn.innerHTML = theme === "dark" ? "☀️ Light Mode" : "🌙 Dark Mode";
  }
}

// Event Listeners
function setupEventListeners() {
  // Domain tab toggles
  document.getElementById("btnTaskFraud").addEventListener("click", () => {
    switchTask("fraud");
  });
  document.getElementById("btnTaskLoan").addEventListener("click", () => {
    switchTask("loan");
  });

  // Slider change
  const slider = document.getElementById("threshSlider");
  slider.addEventListener("input", (e) => {
    onThresholdChange(parseFloat(e.target.value));
  });

  // Copy BibTeX
  const copyBtn = document.getElementById("copyBibtexBtn");
  if (copyBtn) {
    copyBtn.addEventListener("click", () => {
      const bibtex = document.getElementById("bibtexCode").innerText;
      navigator.clipboard.writeText(bibtex).then(() => {
        copyBtn.innerText = "Copied!";
        setTimeout(() => (copyBtn.innerText = "Copy BibTeX"), 2000);
      });
    });
  }
}

// Load precomputed data
async function loadSimulatorData() {
  try {
    const resp = await fetch("assets/data/simulator_data.json");
    if (!resp.ok) throw new Error("Fetch failed");
    simData = await resp.json();
  } catch (err) {
    console.warn("Could not load external json, using embedded data fallback:", err);
    // Embedded minimal fallback
    simData = {
      fraud: {
        optimal_threshold: 0.023,
        cost_matrix: { cost_fn: 200.0, cost_fp: 5.0, cost_tp: 5.0, cost_tn: 0.0 },
        curve: {
          threshold: [0.01, 0.023, 0.05, 0.1, 0.2, 0.5, 0.8],
          cost: [18110, 12160, 14200, 16800, 18900, 21720, 29000],
          recall: [0.946, 0.917, 0.865, 0.812, 0.771, 0.744, 0.62],
          precision: [0.105, 0.23, 0.45, 0.65, 0.78, 0.85, 0.91],
          fn: [9, 14, 23, 32, 39, 43, 64],
          fp: [1361, 516, 210, 78, 35, 22, 10],
          tp: [159, 154, 145, 136, 129, 125, 104],
          tn: [38471, 39316, 39622, 39754, 39797, 39810, 39822]
        }
      },
      loan: {
        optimal_threshold: 0.142,
        cost_matrix: { cost_fn: 5000.0, cost_fp: 500.0, cost_tp: 500.0, cost_tn: 0.0 },
        curve: {
          threshold: [0.05, 0.1, 0.142, 0.2, 0.3, 0.5, 0.7],
          cost: [2900000, 2100000, 1886500, 2400000, 3500000, 7094500, 7500000],
          recall: [0.95, 0.89, 0.854, 0.78, 0.65, 0.086, 0.02],
          precision: [0.22, 0.38, 0.466, 0.54, 0.63, 0.731, 0.82],
          fn: [77, 170, 226, 340, 541, 1414, 1516],
          fp: [5000, 2700, 1513, 1100, 680, 49, 10],
          tp: [1470, 1377, 1321, 1207, 1006, 133, 31],
          tn: [5453, 7753, 8940, 9353, 9773, 10404, 10443]
        }
      }
    };
  }
}

function switchTask(task) {
  currentTask = task;
  document.getElementById("btnTaskFraud").classList.toggle("active", task === "fraud");
  document.getElementById("btnTaskLoan").classList.toggle("active", task === "loan");
  
  const taskObj = simData[task];
  const opt = taskObj.optimal_threshold;
  
  const slider = document.getElementById("threshSlider");
  slider.value = opt.toFixed(3);
  
  document.getElementById("taskDescription").innerText = task === "fraud"
    ? "Task A: Credit Card Fraud Detection (1.85M transactions, 0.58% extreme imbalance, 40:1 FN/FP cost ratio)"
    : "Task B: Loan Default Prediction (252K applicants, 12.30% moderate imbalance, 10:1 FN/FP cost ratio)";
    
  updateSimulatorView();
}

function onThresholdChange(thresh) {
  document.getElementById("threshValBadge").innerText = thresh.toFixed(3);
  updateMetricsForThreshold(thresh);
  if (chartInstance) {
    updateChartMarker(thresh);
  }
}

function updateSimulatorView() {
  const taskObj = simData[currentTask];
  const opt = taskObj.optimal_threshold;
  document.getElementById("threshSlider").value = opt.toFixed(3);
  document.getElementById("threshValBadge").innerText = opt.toFixed(3);
  
  updateMetricsForThreshold(opt);
  renderChart();
}

function updateMetricsForThreshold(thresh) {
  const curve = simData[currentTask].curve;
  
  // Find nearest index
  let bestIdx = 0;
  let minDiff = 999;
  for (let i = 0; i < curve.threshold.length; i++) {
    const diff = Math.abs(curve.threshold[i] - thresh);
    if (diff < minDiff) {
      minDiff = diff;
      bestIdx = i;
    }
  }

  // Cost at 0.50
  let idx50 = 0;
  let minDiff50 = 999;
  for (let i = 0; i < curve.threshold.length; i++) {
    const diff = Math.abs(curve.threshold[i] - 0.50);
    if (diff < minDiff50) {
      minDiff50 = diff;
      idx50 = i;
    }
  }

  const costCurr = curve.cost[bestIdx];
  const cost50 = curve.cost[idx50];
  const savings = cost50 - costCurr;
  const savingsPct = cost50 > 0 ? (savings / cost50) * 100 : 0;
  const recall = curve.recall[bestIdx] * 100;
  const prec = curve.precision[bestIdx] * 100;
  
  document.getElementById("simCostVal").innerText = `$${costCurr.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  document.getElementById("simSavingsVal").innerText = `${savingsPct.toFixed(1)}% ($${savings.toLocaleString("en-US", { maximumFractionDigits: 0 })})`;
  document.getElementById("simRecallVal").innerText = `${recall.toFixed(1)}%`;
  document.getElementById("simPrecisionVal").innerText = `${prec.toFixed(1)}%`;

  document.getElementById("cmTP").innerText = curve.tp[bestIdx].toLocaleString();
  document.getElementById("cmFP").innerText = curve.fp[bestIdx].toLocaleString();
  document.getElementById("cmFN").innerText = curve.fn[bestIdx].toLocaleString();
  document.getElementById("cmTN").innerText = curve.tn[bestIdx].toLocaleString();
}

// Chart.js rendering for Cost vs Threshold curve
function renderChart() {
  const ctx = document.getElementById("costChart");
  if (!ctx) return;

  const isDark = document.documentElement.getAttribute("data-theme") !== "light";
  const textColor = isDark ? "#94a3b8" : "#475569";
  const gridColor = isDark ? "rgba(255, 255, 255, 0.08)" : "rgba(0, 0, 0, 0.06)";
  
  const curve = simData[currentTask].curve;
  const optThresh = simData[currentTask].optimal_threshold;

  if (chartInstance) {
    chartInstance.destroy();
  }

  chartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: curve.threshold.map(t => t.toFixed(3)),
      datasets: [
        {
          label: "Total Financial Cost ($)",
          data: curve.cost,
          borderColor: "#f43f5e",
          backgroundColor: "rgba(244, 63, 94, 0.1)",
          yAxisID: "yCost",
          borderWidth: 2.5,
          tension: 0.2,
          pointRadius: 0,
        },
        {
          label: "Recall (Capture Rate)",
          data: curve.recall.map(r => r * 100),
          borderColor: "#38bdf8",
          yAxisID: "yRate",
          borderWidth: 2,
          borderDash: [4, 4],
          pointRadius: 0,
        },
        {
          label: "Precision",
          data: curve.precision.map(p => p * 100),
          borderColor: "#10b981",
          yAxisID: "yRate",
          borderWidth: 1.5,
          borderDash: [2, 2],
          pointRadius: 0,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false,
      },
      plugins: {
        legend: {
          labels: { color: textColor, font: { family: "sans-serif", weight: "600" } }
        },
        tooltip: {
          backgroundColor: isDark ? "#182234" : "#ffffff",
          titleColor: isDark ? "#ffffff" : "#0f172a",
          bodyColor: isDark ? "#e2e8f0" : "#334155",
          borderColor: isDark ? "#27354f" : "#cbd5e1",
          borderWidth: 1,
        }
      },
      scales: {
        x: {
          title: { display: true, text: "Decision Threshold (p*)", color: textColor },
          ticks: { color: textColor, maxTicksLimit: 12 },
          grid: { color: gridColor },
        },
        yCost: {
          type: "linear",
          position: "left",
          title: { display: true, text: "Total Operational Cost ($)", color: "#f43f5e" },
          ticks: {
            color: "#f43f5e",
            callback: value => "$" + value.toLocaleString()
          },
          grid: { color: gridColor },
        },
        yRate: {
          type: "linear",
          position: "right",
          title: { display: true, text: "Recall & Precision (%)", color: "#38bdf8" },
          min: 0,
          max: 100,
          ticks: {
            color: "#38bdf8",
            callback: value => value + "%"
          },
          grid: { drawOnChartArea: false },
        }
      }
    }
  });
}

function updateChartMarker(thresh) {
  // Chart tooltip/interaction helper
}
