document.addEventListener('DOMContentLoaded', () => {
    const resumeForm = document.getElementById('resume-form');
    const resumeInput = document.getElementById('resume');
    const fileNameDisplay = document.getElementById('file-name');
    const fileButton = document.getElementById('file-button');
    const submitButton = document.getElementById('submit-button');
    const scoreValue = document.getElementById('score-value');
    const resumeText = document.getElementById('resume-text');
    const jobDescription = document.getElementById('job-description');
    const useDefaultJd = document.getElementById('use-default-jd');
    const defaultJdNotice = document.getElementById('default-jd-notice');

    // Analysis elements
    const keywordMatches = document.getElementById('keyword-matches');
    const missingKeywords = document.getElementById('missing-keywords');
    const educationSection = document.getElementById('education-section');
    const experienceSection = document.getElementById('experience-section');
    const skillsSection = document.getElementById('skills-section');

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

    // Display resume section content
    function displaySectionContent(sectionElement, sectionData) {
        sectionElement.innerHTML = '';

        if (sectionData && sectionData.length > 0) {
            sectionData.forEach(item => {
                const paragraph = document.createElement('p');
                paragraph.textContent = item;
                sectionElement.appendChild(paragraph);
            });
        } else {
            const paragraph = document.createElement('p');
            paragraph.textContent = 'No information found';
            sectionElement.appendChild(paragraph);
        }
    }

    // Display keyword matches
    function displayKeywords(container, keywords, isMatch) {
        container.innerHTML = '';

        if (keywords && keywords.length > 0) {
            keywords.forEach(keyword => {
                const keywordElement = document.createElement('span');
                keywordElement.textContent = keyword;
                keywordElement.className = `keyword ${isMatch ? 'match' : 'missing'}`;
                container.appendChild(keywordElement);
            });
        } else {
            const message = document.createElement('p');
            message.textContent = isMatch ? 'No matching keywords' : 'No missing keywords';
            message.style.color = '#888';
            container.appendChild(message);
        }
    }

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
        defaultJdNotice.classList.add('hidden');

        // Clear previous analysis
        keywordMatches.innerHTML = '';
        missingKeywords.innerHTML = '';
        educationSection.innerHTML = '';
        experienceSection.innerHTML = '';
        skillsSection.innerHTML = '';

        // Create form data
        const formData = new FormData();
        formData.append('resume', resumeInput.files[0]);

        // Add job description if provided
        if (jobDescription.value.trim()) {
            formData.append('job_description', jobDescription.value.trim());
        }

        // Add use_default_jd flag
        formData.append('use_default_job_desc', useDefaultJd.checked);

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

            // Set color based on score
            const scoreCircle = document.querySelector('.score-circle');
            if (scorePercentage >= 70) {
                scoreCircle.style.borderColor = '#4CAF50'; // Green for high score
            } else if (scorePercentage >= 40) {
                scoreCircle.style.borderColor = '#FFC107'; // Yellow/amber for medium score
            } else {
                scoreCircle.style.borderColor = '#F44336'; // Red for low score
            }

            scoreValue.textContent = `${scorePercentage}%`;

            // Show default JD notice if applicable
            if (data.used_default_jd) {
                defaultJdNotice.classList.remove('hidden');
                defaultJdNotice.textContent = 'Using default job description';
            } else if (!jobDescription.value.trim() && !useDefaultJd.checked) {
                defaultJdNotice.classList.remove('hidden');
                defaultJdNotice.textContent = 'No job description provided - score is 0%';
            } else {
                defaultJdNotice.classList.add('hidden');
            }

            // Update resume text preview
            resumeText.textContent = data.resume_text;

            // Display keyword matches
            if (data.match_details && data.match_details.keywords) {
                const matchedKeywords = Object.keys(data.match_details.keywords)
                    .filter(keyword => data.match_details.keywords[keyword]);
                displayKeywords(keywordMatches, matchedKeywords, true);
            }

            // Display missing keywords
            if (data.match_details && data.match_details.missing_keywords) {
                displayKeywords(missingKeywords, data.match_details.missing_keywords, false);
            }

            // Display resume sections
            if (data.match_details && data.match_details.sections) {
                displaySectionContent(educationSection, data.match_details.sections.education);
                displaySectionContent(experienceSection, data.match_details.sections.experience);
                displaySectionContent(skillsSection, data.match_details.sections.skills);
            }

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