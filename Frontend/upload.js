const galleryInput = document.getElementById("galleryFile");
const cameraInput = document.getElementById("cameraFile");
const fileStatus = document.getElementById("fileStatus");

function updateFileStatus(fileInput) {
    if (fileInput.files.length > 0) {
        fileStatus.textContent = `Selected: ${fileInput.files[0].name}`;
    } else {
        fileStatus.textContent = "No image selected";
    }
}

galleryInput.addEventListener("change", () => {
    cameraInput.value = ""; // Clear camera file if gallery chosen
    updateFileStatus(galleryInput);
});

cameraInput.addEventListener("change", () => {
    galleryInput.value = ""; // Clear gallery file if camera chosen
    updateFileStatus(cameraInput);
});

document.getElementById("uploadForm").addEventListener("submit", async function(event) {
    event.preventDefault();

    const button = document.querySelector(".predict-btn");
    button.disabled = true;
    button.textContent = "Processing...";

    let fileInput = galleryInput.files[0] || cameraInput.files[0];

    if (!fileInput) {
        alert("Please select or take a photo first!");

        button.disabled = false;
        button.textContent = "Predict";

        return;
    }

    fileStatus.textContent = "Uploading & Processing...";

    const formData = new FormData();
    formData.append("file", fileInput);

    try {
        const response = await fetch("https://plant-disease-prediction-deployment.onrender.com/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        fileStatus.textContent = "Done!";

        button.disabled = false;
        button.textContent = "Predict";

        console.log("FULL API RESPONSE:", data);
        // TEMPORARY DELAY
        setTimeout(() => {
            window.location.href = "results.html";
        }, 5000);

        localStorage.setItem("status", data.status);
        localStorage.setItem("disease", data.disease);
        const confidence = parseFloat(data.confidence);

        localStorage.setItem("accuracy",
            !isNaN(confidence) ? confidence : 0
        );

        //window.location.href = "results.html";
    } catch (err) {
        console.error("Upload failed", err);
        button.disabled = false;
        button.textContent = "Predict";
    }
});

