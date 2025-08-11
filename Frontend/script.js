document.getElementById("uploadForm").addEventListener("submit", async function(event) {
    event.preventDefault();

    const fileInput = document.getElementById("imageFile");
    if (!fileInput.files.length) {
        alert("Please select an image first!");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        document.getElementById("status").textContent = data.status || "Error";
        document.getElementById("disease").textContent = data.disease || "-";
        document.getElementById("accuracy").textContent = 
            data.accuracy !== "-" ? data.accuracy.toFixed(2) : "-";

        document.getElementById("result").classList.remove("hidden");
    } catch (error) {
        document.getElementById("status").textContent = "Error";
        document.getElementById("disease").textContent = "-";
        document.getElementById("accuracy").textContent = "-";
        document.getElementById("result").classList.remove("hidden");
        console.error(error);
    }
});
