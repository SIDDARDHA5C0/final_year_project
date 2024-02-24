
var socket = io();  // Initialize Socket.IO client

document.getElementById('uploadForm').addEventListener('submit', function (event) {
    event.preventDefault();
    var formData = new FormData(this);

    fetch("/video_feed", {
        method: 'POST',
        body: formData
    })
        .then(response => {
            console.log("Video uploaded successfully");
        })
        .catch(error => {
            console.error('Error uploading video:', error);
        });
});

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
        console.log(data)
        const videoUrl = data.video_url;
        console.log(videoUrl)
        // Set the video source
        if (videoUrl === "no") {
            document.getElementById('myVideo').innerHTML = `<p>video not found</p>`;
        } else {
            document.getElementById('myVideo').src = "/static/output.mp4";
            document.getElementById('myVideo').style.display = 'block';
        }
    } catch (error) {
        console.error('Error fetching video URL:', error);
    }
});
