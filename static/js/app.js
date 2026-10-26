// Small progressive enhancements; every page also works without JavaScript.
document.addEventListener("DOMContentLoaded", () => {
  // Show loading feedback and prevent double submissions.
  document.querySelectorAll("form.needs-loading").forEach((form) => {
    form.addEventListener("submit", () => {
      const button = form.querySelector("button[type=submit][data-loading-text]");
      if (!button) return;
      button.disabled = true;
      const spinner = document.createElement("span");
      spinner.className = "spinner-border spinner-border-sm me-2";
      spinner.setAttribute("aria-hidden", "true");
      button.textContent = button.dataset.loadingText;
      button.prepend(spinner);
    });
  });

  // Ask before destructive actions.
  document.querySelectorAll("[data-confirm]").forEach((element) => {
    element.addEventListener("click", (event) => {
      if (!window.confirm(element.dataset.confirm)) event.preventDefault();
    });
  });

  // "Select all" / "Select none" helpers for checkbox groups.
  document.querySelectorAll("[data-check-all], [data-uncheck-all]").forEach((button) => {
    button.addEventListener("click", () => {
      const name = button.dataset.checkAll || button.dataset.uncheckAll;
      const checked = Boolean(button.dataset.checkAll);
      document.querySelectorAll(`input[type=checkbox][name="${name}"]`).forEach((box) => {
        box.checked = checked;
      });
    });
  });
});
