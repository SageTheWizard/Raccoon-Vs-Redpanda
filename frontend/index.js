const form = document.getElementById('uploadForm');
const resultElement = document.getElementById('result');

form.addEventListener("submit", e => {
    e.preventDefault();
    const fileInput = document.getElementById('uploadFile');

    if(fileInput.files.length)
    {
        const file = fileInput.files[0];

        const form = new FormData();
        form.append("picofRacoon", file, file.name);

        const settings = {
            "url": "http://localhost:5000/upload",
            "method": "POST",
            "timeout": 0,
            "processData": false,
            "mimeType": "multipart/form-data",
            "contentType": false,
            "data": form
        };

        // $ is JQUERY jank
        $.ajax(settings).done(function (response) {
            // resultElement.innerHTML = response ? "Raccoon" : "Panda Boi";
            // Do something with the response here
            resultElement.innerHTML = response;
        });
    }
});