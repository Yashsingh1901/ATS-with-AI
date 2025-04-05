document.addEventListener('DOMContentLoaded', () => {
    const resumeForm = document.getElementById('resume-form');
    const resumeInput = document.getElementById('resume');
    const fileNameDisplay = document.getElementById('file-name');
    const fileButton = document.getElementById('file-button');
    const submitButton = document.getElementById('submit-button');
    const scoreValue = document.getElementById('score-value');
    const resumeText = document.getElementById('resume-text');
    const jobDescription = document.getElementById('job-description');

    // Update file name display when a file is selected
    resumeInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            fileNameDisplay.textContent = e.target.files[0].name;
        } else {
            fileNameDisplay.textContent = 'No file chosen';
        }
    });

    // Trigger file input when custom button is clicked
    fileButton.addEventListener('click', () => {
        resumeInput.click();
    });

    // Handle form submission
    resumeForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Validate form
        if (!resumeInput.files[0]) {
            alert('Please select a resume file.');
            return;
        }

        // Set loading state
        submitButton.disabled = true;
        submitButton.classList.add('loading');
        scoreValue.textContent = 'Calculating...';
        resumeText.textContent = 'Processing resume...';

        // Create form data
        const formData = new FormData();
        formData.append('resume', resumeInput.files[0]);

        // Add job description if provided
        if (jobDescription.value.trim()) {
            formData.append('job_description', jobDescription.value.trim());
        }

        try {
            // API endpoint
            const apiUrl = 'http://localhost:8000/rank-resume';
            console.log('Sending request to:', apiUrl);

            // Make API call with detailed error logging
            const response = await fetch(apiUrl, {
                method: 'POST',
                body: formData,
                // Don't set Content-Type header - browser will set it with boundary
                headers: {
                    // No specific headers for multipart/form-data
                }
            });

            console.log('Response status:', response.status);

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Error response:', errorText);
                throw new Error(`Server error (${response.status}): ${errorText || 'Unknown error'}`);
            }

            const data = await response.json();
            console.log('Received data:', data);

            // Update score (convert to percentage)
            const scorePercentage = Math.round(data.score * 100);
            scoreValue.textContent = `${scorePercentage}%`;

            // Update resume text preview
            resumeText.textContent = data.resume_text;

            // Scroll to results if on mobile
            if (window.innerWidth <= 768) {
                document.getElementById('results-container').scrollIntoView({
                    behavior: 'smooth'
                });
            }

        } catch (error) {
            console.error('Error:', error);
            scoreValue.textContent = 'Error';
            resumeText.textContent = `Failed to process resume: ${error.message}. Please try again.`;
            alert(`Error processing resume: ${error.message}`);
        } finally {
            // Remove loading state
            submitButton.disabled = false;
            submitButton.classList.remove('loading');
        }
    });
}); 