// Basic front-end logic: AJAX submit for predict, incare forms and simple UI handling.

document.addEventListener("DOMContentLoaded", () => {
  // Predict form (AJAX)
  const predictForm = document.getElementById("predictForm");
  if (predictForm) {
    predictForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const formData = new FormData(predictForm);
      const respBox = document.getElementById("resultBox");
      const pre = document.getElementById("resultPre");
      respBox.classList.add("hidden");
      try {
        const res = await fetch("/predict", { method: "POST", body: formData });
        const data = await res.json();
        if (res.ok) {
          pre.textContent = JSON.stringify(data, null, 2);
          respBox.classList.remove("hidden");
          window.scrollTo({ top: respBox.offsetTop - 80, behavior: "smooth" });
        } else {
          pre.textContent = JSON.stringify(data, null, 2);
          respBox.classList.remove("hidden");
        }
      } catch (err) {
        pre.textContent = "Request error: " + err;
        respBox.classList.remove("hidden");
      }
    });
  }

  // Reset button
  const resetBtn = document.getElementById("resetBtn");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      predictForm.reset();
      document.getElementById("resultBox").classList.add("hidden");
    });
  }

  // InCare individual form
  const individualForm = document.getElementById("individualForm");
  if (individualForm) {
    individualForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const frm = new FormData(individualForm);
      try {
        const res = await fetch("/incare-outcare", { method: "POST", body: frm });
        const data = await res.json();
        const indResult = document.getElementById("indResult");
        indResult.textContent = JSON.stringify(data, null, 2);
        indResult.classList.remove("hidden");
      } catch (err) {
        alert("Error: " + err);
      }
    });
    document.getElementById("clearInd").addEventListener("click", () => {
      individualForm.reset();
      document.getElementById("indResult").classList.add("hidden");
    });
  }

  // CSV batch form
  const csvForm = document.getElementById("csvForm");
  if (csvForm) {
    csvForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const frm = new FormData(csvForm);
      try {
        const res = await fetch("/incare-outcare", { method: "POST", body: frm });
        const data = await res.json();
        const csvResult = document.getElementById("csvResult");
        if (res.ok) {
          csvResult.textContent = JSON.stringify(data, null, 2);
          csvResult.classList.remove("hidden");
        } else {
          csvResult.textContent = "Error: " + JSON.stringify(data, null, 2);
          csvResult.classList.remove("hidden");
        }
      } catch (err) {
        alert("Error: " + err);
      }
    });
  }
});
