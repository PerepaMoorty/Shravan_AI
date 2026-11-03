using System.Collections.Generic;
using System.IO;

namespace ShravanAI.Playback
{
    /// <summary>
    /// Deliberately minimal CSV reader for the pipeline's own exports only
    /// (numeric columns + single-token strings, no quoting/embedded commas).
    /// Not a general-purpose CSV parser — don't reuse it for arbitrary files.
    /// </summary>
    public static class MiniCsv
    {
        public static List<Dictionary<string, string>> ReadRows(string path)
        {
            var rows = new List<Dictionary<string, string>>();
            using var reader = new StreamReader(path);

            string headerLine = reader.ReadLine();
            if (headerLine == null) return rows;
            string[] headers = headerLine.Split(',');

            string line;
            while ((line = reader.ReadLine()) != null)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;
                string[] values = line.Split(',');
                var row = new Dictionary<string, string>();
                for (int i = 0; i < headers.Length && i < values.Length; i++)
                {
                    row[headers[i]] = values[i];
                }
                rows.Add(row);
            }
            return rows;
        }
    }
}
