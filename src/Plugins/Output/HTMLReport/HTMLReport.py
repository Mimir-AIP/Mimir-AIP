import os


class HTMLReportGenerator:
    def __init__(self, output_directory="reports"):
        self.output_directory = output_directory
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_directory, exist_ok=True)


    def generate_report(self, title, sections, filename="report.html"):
        """
        Generate an HTML report with multiple text and JavaScript sections.

        :param title: Title of the HTML document
        :param sections: List of sections, where each section is a dictionary with:
                         - "heading": Heading for the section
                         - "text": Text content for the section (HTML allowed)
                         - "javascript": JavaScript code for the section
        :param filename: Name of the output HTML file
        """
        # Generate HTML content for all text and JavaScript sections
        section_html = ""
        for i, section in enumerate(sections):
            heading = section.get("heading", f"Section {i + 1}")
            text = section.get("text", "")
            javascript = section.get("javascript", "")

            # Add section content to the report
            section_html += f"""
            <div class="section">
                <h2>{heading}</h2>
                <div class="text-content">
                    {text}
                </div>
                <script>
                    {javascript}
                </script>
            </div>
            """

        # Full HTML template
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="leaflet.css" />
    <script src="leaflet.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 2em;
            color: #333;
            background-color: #f5f5f5;
        }}
        h1 {{
            text-align: center;
            color: #444;
        }}
        .section {{
            margin-bottom: 20px;
            padding: 15px;
            border: 1px solid #ccc;
            border-radius: 10px;
            background-color: #fff;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .section h2 {{
            color: #555;
            border-bottom: 2px solid #007BFF;
            margin-bottom: 15px;
            padding-bottom: 5px;
        }}
        .text-content {{
            padding: 10px;
            background-color: #fcfcfc;
            border: 1px solid #eee;
            border-radius: 5px;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        footer {{
            text-align: center;
            margin-top: 20px;
            font-size: 0.9em;
            color: #888;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    {section_html}
    <footer>
        Generated by HTMLReportGenerator
    </footer>
</body>
</html>
"""
        # Full path for the output file
        report_path = os.path.join(self.output_directory, filename)

        # Write the HTML content to the file
        with open(report_path, "w", encoding="utf-8") as file:
            file.write(html_template)

        print(f"Report generated: {report_path}")
        return report_path


# Usage Example
if __name__ == "__main__":
    generator = HTMLReportGenerator()

    # Define sections with headings, text, and JavaScript
    #TODO add sections with leafletjs maps generated by existing plugin
    sections = [
        {
            "heading": "Introduction",
            "text": "<p>Welcome to the sample report. This is the introduction.</p>",
            "javascript": "console.log('Intro section loaded!');"
        },
        {
            "heading": "Dynamic Content",
            "text": """
            <p>This section demonstrates the inclusion of <b>dynamic content</b>.</p>
            <p>Below is a counter that updates when you click the button:</p>
            <button onclick="incrementCounter()">Click Me</button>
            <p>Count: <span id="counter">0</span></p>
            """,
            "javascript": """
            let count = 0;
            function incrementCounter() {
                count++;
                document.getElementById('counter').textContent = count;
            }
            """
        },
        {
            "heading": "Map example",
            "text": '<div id="map" style="width: 600px; height: 400px;"></div>',
            "javascript": """
            var map = L.map('map').setView([51.5074, -0.1278], 2);

            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            }).addTo(map);"""
        },
        {
            "heading": "Conclusion",
            "text": "<p>This is the conclusion of the report. Thank you for reading!</p>",
            "javascript": "console.log('Conclusion section loaded!');"
        }
    ]

    # Generate the report
    generator.generate_report(
        title="Sample HTML Report with Multiple Sections",
        sections=sections,
        filename="multi_section_report.html"
    )