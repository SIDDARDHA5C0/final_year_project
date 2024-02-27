
var socket = io();  // Initialize Socket.IO client

// Global variables for frame coordinates
let frameStartX = -1;
let frameStartY = -1;
let frameEndX = -1;
let frameEndY = -1;
let fframe = '';
// Function to handle the upload form submission
document.getElementById('uploadForm').addEventListener('submit', function (event) {
    event.preventDefault();
    var formData = new FormData(this);

    fetch("/upload_video", {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            console.log("Frame dimensions:", data.frame_width, "x", data.frame_height);
            const canvas = document.getElementById('frameCanvas');
            const ctx = canvas.getContext('2d');
            fframe = data.frame_path;
            let startX = -1;
            let startY = -1;
            let endX = -1;
            let endY = -1;
            // Create a new image element for the background
            const backgroundImage = new Image();
            backgroundImage.onload = function () {
                // Set canvas dimensions
                const canvasWidth = canvas.offsetWidth;
                const canvasHeight = canvas.offsetHeight;
                canvas.width = canvasWidth;
                canvas.height = canvasHeight;

                // Set canvas background image
                ctx.drawImage(backgroundImage, 0, 0, canvasWidth, canvasHeight);

                // Calculate scaling factors
                const scaleX = data.frame_width / canvasWidth;
                const scaleY = data.frame_height / canvasHeight;

                // Variables to store line coordinates
                let isDrawing = false; // Flag to track drawing state

                canvas.addEventListener('mousedown', onMouseDown);
                canvas.addEventListener('mousemove', onMouseMove);
                canvas.addEventListener('mouseup', onMouseUp);

                function onMouseDown(e) {
                    const rect = e.target.getBoundingClientRect();
                    startX = e.clientX - rect.left;
                    startY = e.clientY - rect.top;
                    isDrawing = true; // Start drawing
                }

                function onMouseMove(e) {
                    if (!isDrawing) {
                        return;
                    }
                    const rect = e.target.getBoundingClientRect();
                    endX = e.clientX - rect.left;
                    endY = e.clientY - rect.top;

                    // Clear canvas
                    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

                    // Redraw background image
                    ctx.drawImage(backgroundImage, 0, 0, canvasWidth, canvasHeight);

                    // Draw line
                    ctx.beginPath();
                    ctx.moveTo(startX, startY);
                    ctx.lineTo(endX, endY);
                    ctx.strokeStyle = 'red'; // Change color as needed
                    ctx.lineWidth = 2; // Change width as needed
                    ctx.stroke();
                }

                function onMouseUp() {
                    isDrawing = false; // Stop drawing

                    // Calculate coordinates relative to the frame
                    frameStartX = startX * scaleX;
                    frameStartY = startY * scaleY;
                    frameEndX = endX * scaleX;
                    frameEndY = endY * scaleY;

                    console.log('Start:', frameStartX, frameStartY);
                    console.log('End:', frameEndX, frameEndY);
                }
            };

            // Set the src attribute to start loading the image
            backgroundImage.src = fframe;
        })
        .catch(error => {
            console.error('Error uploading video:', error);
        });
});

// Function to erase lines and reset coordinates
function eraseLines() {
    // Clear canvas
    const canvas = document.getElementById('frameCanvas');
    const ctx = canvas.getContext('2d');
    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    // Set background image again
    const img = new Image();
    img.src = fframe;
    img.onload = function () {
        ctx.drawImage(img, 0, 0, canvasWidth, canvasHeight);
    };

    // Reset coordinates
    startX = -1;
    startY = -1;
    endX = -1;
    endY = -1;

    console.log("done");
}

// Function to extract coordinates and send them to Flask
function sendCoordinatesToFlask() {
    // Check if any coordinate is still -1
    if (frameStartX === -1 || frameStartY === -1 || frameEndX === -1 || frameEndY === -1) {
        // Raise alert if any coordinate is still -1
        alert('Please draw a line first to get coordinates.');
        return;
    }

    // Send HTTP request to Flask server
    fetch("/video_feed", {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            startX: frameStartX,
            startY: frameStartY,
            endX: frameEndX,
            endY: frameEndY
        })
    })
        .then(response => {
            // Handle response
            console.log('Coordinates sent to Flask');
        })
        .catch(error => {
            console.error('Error sending coordinates to Flask:', error);
        });
}

// Function to fetch the value of counter_A from the server and update the webpage
async function updateCounterA() {
    try {
        const response = await fetch("/counter_A");
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        document.getElementById("counterValue").textContent = data.counter_A;
    } catch (error) {
        console.error('Error fetching counter_A:', error);
    }
}

// Call the function initially and then every 1 second
updateCounterA();
setInterval(updateCounterA, 1000);








socket.on('frame', function (frameBase64) {
    console.log('Frame received');
    var img = document.createElement('img');
    img.src = 'data:image/jpeg;base64,' + frameBase64;
    document.getElementById('video-display').innerHTML = ''; // Clear previous frames
    document.getElementById('video-display').appendChild(img);
});
document.getElementById('playButton').addEventListener('click', async () => {
    try {
        const response = await fetch('/get_video_url');
        const data = await response.json();
        const videoUrl = data.video_url;

        // Set the video source
        const videoElement = document.getElementById('myVideo');
        if (videoUrl === "no") {
            videoElement.innerHTML = `<p>Video not found</p>`;
        } else {

            videoElement.style.display = 'block';
            videoElement.src = videoUrl;
            console.log("video displaying");
            console.log(videoUrl);
        }
    } catch (error) {
        console.error('Error fetching video URL:', error);
    }
});
