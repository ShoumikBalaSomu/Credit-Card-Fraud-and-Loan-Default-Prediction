// Interactive Logic for GitHub Pages ML Showcase

let currentTask = "fraud"; // 'fraud' or 'loan'
let simData = null;
let chartInstance = null;

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", async () => {
  initTheme();
  setupEventListeners();
  if (document.getElementById("threshSlider") || document.getElementById("costChart")) {
    await loadSimulatorData();
    updateSimulatorView();
  }
});

// Theme Management
function initTheme() {
  const savedTheme = localStorage.getItem("finrisk-theme") || "dark";
  document.documentElement.setAttribute("data-theme", savedTheme);
  updateThemeButton(savedTheme);

  const toggleBtn = document.getElementById("themeToggleBtn");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const current = document.documentElement.getAttribute("data-theme");
      const next = current === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem("finrisk-theme", next);
      updateThemeButton(next);
      if (chartInstance) renderChart();
    });
  }
}

function updateThemeButton(theme) {
  const btn = document.getElementById("themeToggleBtn");
  if (btn) {
    btn.innerHTML = theme === "dark" ? "☀️ Light Mode" : "🌙 Dark Mode";
  }
}

// Event Listeners
function setupEventListeners() {
  const btnFraud = document.getElementById("btnTaskFraud");
  if (btnFraud) {
    btnFraud.addEventListener("click", () => switchTask("fraud"));
  }
  const btnLoan = document.getElementById("btnTaskLoan");
  if (btnLoan) {
    btnLoan.addEventListener("click", () => switchTask("loan"));
  }

  // Slider change
  const slider = document.getElementById("threshSlider");
  if (slider) {
    slider.addEventListener("input", (e) => {
      onThresholdChange(parseFloat(e.target.value));
    });
  }

  // Preset buttons if present
  const btnOpt = document.getElementById("btnPresetOptimal");
  if (btnOpt) {
    btnOpt.addEventListener("click", () => {
      if (simData && simData[currentTask]) {
        const opt = simData[currentTask].optimal_threshold;
        if (slider) slider.value = opt.toFixed(3);
        onThresholdChange(opt);
      }
    });
  }

  const btnDefault = document.getElementById("btnPresetDefault");
  if (btnDefault) {
    btnDefault.addEventListener("click", () => {
      if (slider) slider.value = (0.50).toFixed(3);
      onThresholdChange(0.50);
    });
  }

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

  // Copy APA
  const copyApaBtn = document.getElementById("copyApaBtn");
  if (copyApaBtn) {
    copyApaBtn.addEventListener("click", () => {
      const apa = document.getElementById("apaCode").innerText;
      navigator.clipboard.writeText(apa).then(() => {
        copyApaBtn.innerText = "Copied!";
        setTimeout(() => (copyApaBtn.innerText = "Copy APA"), 2000);
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
        optimal_threshold: 0.115,
        cost_matrix: { cost_fn: 200.0, cost_fp: 5.0, cost_tp: 0.0, cost_tn: 0.0 },
        curve: {
          threshold: [0.01, 0.05, 0.115, 0.2, 0.3, 0.5, 0.8],
          cost: [18110, 14200, 14980, 16800, 18900, 23440, 29000],
          recall: [0.946, 0.917, 0.865, 0.812, 0.771, 0.744, 0.62],
          precision: [0.105, 0.23, 0.45, 0.65, 0.78, 0.85, 0.91],
          fn: [9, 14, 23, 32, 39, 43, 64],
          fp: [1361, 516, 210, 78, 35, 22, 10],
          tp: [159, 154, 145, 136, 129, 125, 104],
          tn: [38471, 39316, 39622, 39754, 39797, 39810, 39822]
        }
      },
      loan: {
        optimal_threshold: 0.135,
        cost_matrix: { cost_fn: 5000.0, cost_fp: 500.0, cost_tp: 0.0, cost_tn: 0.0 },
        curve: {
          threshold: [0.05, 0.1, 0.135, 0.2, 0.3, 0.5, 0.7],
          cost: [2900000, 2100000, 1938500, 2400000, 3500000, 7094500, 7500000],
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
  const btnFraud = document.getElementById("btnTaskFraud");
  const btnLoan = document.getElementById("btnTaskLoan");
  if (btnFraud) btnFraud.classList.toggle("active", task === "fraud");
  if (btnLoan) btnLoan.classList.toggle("active", task === "loan");
  
  // Switch benchmarks tables if on benchmarks page
  const bFraud = document.getElementById("benchmarksFraud");
  const bLoan = document.getElementById("benchmarksLoan");
  if (bFraud && bLoan) {
    bFraud.style.display = task === "fraud" ? "block" : "none";
    bLoan.style.display = task === "loan" ? "block" : "none";
  }

  if (simData && simData[task]) {
    const taskObj = simData[task];
    const opt = taskObj.optimal_threshold;
    
    const slider = document.getElementById("threshSlider");
    if (slider) slider.value = opt.toFixed(3);
    
    const desc = document.getElementById("taskDescription");
    if (desc) {
      desc.innerText = task === "fraud"
        ? "Task A: Credit Card Fraud Detection (1.85M transactions, 0.58% extreme imbalance, 40:1 FN/FP cost ratio)"
        : "Task B: Loan Default Prediction (252K applicants, 12.30% moderate imbalance, 10:1 FN/FP cost ratio)";
    }
      
    updateSimulatorView();
  }
}

function onThresholdChange(thresh) {
  const badge = document.getElementById("threshValBadge");
  if (badge) badge.innerText = thresh.toFixed(3);
  updateMetricsForThreshold(thresh);
  if (chartInstance) {
    updateChartMarker(thresh);
  }
}

function updateSimulatorView() {
  if (!simData || !simData[currentTask]) return;
  const taskObj = simData[currentTask];
  const opt = taskObj.optimal_threshold;
  const slider = document.getElementById("threshSlider");
  if (slider) slider.value = opt.toFixed(3);
  const badge = document.getElementById("threshValBadge");
  if (badge) badge.innerText = opt.toFixed(3);
  
  updateMetricsForThreshold(opt);
  renderChart();
}

function updateMetricsForThreshold(thresh) {
  if (!simData || !simData[currentTask]) return;
  const curve = simData[currentTask].curve;
  if (!curve || !curve.threshold) return;
  
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
  const savings = Math.max(0, cost50 - costCurr);
  const savingsPct = cost50 > 0 ? (savings / cost50) * 100 : 0;
  const recall = (curve.recall[bestIdx] || 0) * 100;
  const prec = (curve.precision[bestIdx] || 0) * 100;
  
  const elCost = document.getElementById("simCostVal");
  if (elCost) elCost.innerText = `$${costCurr.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  
  const elSav = document.getElementById("simSavingsVal");
  if (elSav) elSav.innerText = `${savingsPct.toFixed(1)}% ($${savings.toLocaleString("en-US", { maximumFractionDigits: 0 })})`;
  
  const elRec = document.getElementById("simRecallVal");
  if (elRec) elRec.innerText = `${recall.toFixed(1)}%`;
  
  const elPrec = document.getElementById("simPrecisionVal");
  if (elPrec) elPrec.innerText = `${prec.toFixed(1)}%`;

  const elTP = document.getElementById("cmTP");
  if (elTP) elTP.innerText = curve.tp[bestIdx].toLocaleString();
  const elFP = document.getElementById("cmFP");
  if (elFP) elFP.innerText = curve.fp[bestIdx].toLocaleString();
  const elFN = document.getElementById("cmFN");
  if (elFN) elFN.innerText = curve.fn[bestIdx].toLocaleString();
  const elTN = document.getElementById("cmTN");
  if (elTN) elTN.innerText = curve.tn[bestIdx].toLocaleString();
}

// Chart.js rendering for Cost vs Threshold curve
function renderChart() {
  const ctx = document.getElementById("costChart");
  if (!ctx || !simData || !simData[currentTask]) return;

  const isDark = document.documentElement.getAttribute("data-theme") !== "light";
  const textColor = isDark ? "#94a3b8" : "#475569";
  const gridColor = isDark ? "rgba(255, 255, 255, 0.08)" : "rgba(0, 0, 0, 0.06)";
  
  const curve = simData[currentTask].curve;
  if (!curve || !curve.threshold) return;

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
          label: "Recall (Capture Rate %)",
          data: curve.recall.map(r => r * 100),
          borderColor: "#38bdf8",
          yAxisID: "yRate",
          borderWidth: 2,
          borderDash: [4, 4],
          pointRadius: 0,
        },
        {
          label: "Precision (%)",
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
