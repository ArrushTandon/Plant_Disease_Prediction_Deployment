/*
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

        // Store in localStorage
        localStorage.setItem("status", response.ok ? "Success" : "Error");
        localStorage.setItem("disease", data.disease || "-");
        localStorage.setItem("accuracy", data.accuracy ? data.accuracy.toFixed(2) : "-");

        // Redirect to results page
        window.location.href = "results.html";
    } catch (error) {
        localStorage.setItem("status", "Error");
        localStorage.setItem("disease", "-");
        localStorage.setItem("accuracy", "-");
        window.location.href = "results.html";
    }
});
*/
/*
document.getElementById("uploadForm").addEventListener("submit", async function(event) {
    event.preventDefault();

    let fileInput = document.getElementById("galleryFile").files[0] 
                 || document.getElementById("cameraFile").files[0];

    if (!fileInput) {
        alert("Please select or take a photo first!");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput);

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        console.log(data);

        localStorage.setItem("status", data.status);
        localStorage.setItem("disease", data.disease);
        localStorage.setItem("accuracy", data.accuracy);
        window.location.href = "results.html";
    } catch (err) {
        console.error("Upload failed", err);
    }
});
*/
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

galleryInput.addEventListener("change", () => updateFileStatus(galleryInput));
cameraInput.addEventListener("change", () => updateFileStatus(cameraInput));

document.getElementById("uploadForm").addEventListener("submit", async function(event) {
    event.preventDefault();

    let fileInput = galleryInput.files[0] || cameraInput.files[0];

    if (!fileInput) {
        alert("Please select or take a photo first!");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput);

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        console.log(data);

        localStorage.setItem("status", data.status);
        localStorage.setItem("disease", data.disease);
        localStorage.setItem("accuracy", data.accuracy);
        window.location.href = "results.html";
    } catch (err) {
        console.error("Upload failed", err);
    }
});

