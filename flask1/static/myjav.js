document.getElementById('uploadForm').addEventListener('submit', function (event) {
    event.preventDefault();
    const formData = new FormData(this);
    fetch("/video_feed", {
        method: 'POST',
        body: formData
    })
        .then(response => {

            function handleFrame(frameUrl) {
                var img = document.createElement('img');
                img.src = frameUrl;
                img.style.maxWidth = '100%';
                img.style.maxHeight = '100%';
                var videoDiv = document.getElementById('video-display');
                while (videoDiv.firstChild) {
                    videoDiv.removeChild(videoDiv.firstChild);
                }
                videoDiv.appendChild(img);
            }
            function processResponse(response) {
                var reader = response.body.getReader();
                async function read() {
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) {
                            break;
                        }
                        // Convert Uint8Array to string
                        //const frameString = new TextDecoder().decode(value);

                        // Find the index of the start of the frame data after the header
                        //const startIndex = frameString.indexOf('\r\n\r\n') + 4;

                        // Find the index of the end of the frame data before the next occurrence of '\r\n\r\n'
                        //const endIndex = frameString.lastIndexOf('\r\n');

                        // Extract the frame data from the byte string
                        //const frameData = value.subarray(startIndex, endIndex);
                        console.log(value)
                        // Create a Blob object from the frame data
                        var blob = new Blob([value], { type: 'jpg' }); // Adjust the type as needed

                        // Create object URL from Blob
                        var imageUrl = URL.createObjectURL(blob);

                        // Handle the image URL
                        handleFrame(imageUrl);
                    }
                }
                read();
            }

            processResponse(response);
        })
        .catch(error => {
            console.error('Error:', error);
        });
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
