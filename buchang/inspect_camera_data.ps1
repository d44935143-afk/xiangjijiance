Add-Type -AssemblyName System.IO.Compression.FileSystem
$filePath = Join-Path $PSScriptRoot 'camera_data/mubiao1_targets_2880.xlsx'
$archive = [IO.Compression.ZipFile]::OpenRead($filePath)
function Read-XmlEntry($entryName) {
    $reader = [IO.StreamReader]::new($archive.GetEntry($entryName).Open())
    try { [xml]$reader.ReadToEnd() } finally { $reader.Dispose() }
}
function Summarize($values) {
    $stats = $values | Measure-Object -Minimum -Maximum -Average
    $avg = $stats.Average
    $sum = 0.0
    foreach ($v in $values) { $sum += ($v - $avg) * ($v - $avg) }
    [ordered]@{count=$values.Count; min=$stats.Minimum; max=$stats.Maximum; mean=$avg; variance=($sum / ($values.Count-1)); std=[Math]::Sqrt($sum / ($values.Count-1))}
}
try {
    $book = Read-XmlEntry 'xl/workbook.xml'
    $relations = Read-XmlEntry 'xl/_rels/workbook.xml.rels'
    foreach ($sheet in $book.workbook.sheets.sheet) {
        $rid = $sheet.GetAttribute('id', 'http://schemas.openxmlformats.org/officeDocument/2006/relationships')
        $target = ($relations.Relationships.Relationship | Where-Object Id -EQ $rid).Target
        $entryPath = if ($target.StartsWith('/')) { $target.TrimStart('/') } else { 'xl/' + $target }
        $xml = Read-XmlEntry $entryPath
        $rows = @($xml.worksheet.sheetData.row)
        $columns = @{}
        $nonNumeric = 0
        foreach ($row in $rows) {
            foreach ($cell in $row.c) {
                $key = $cell.r -replace '\d',''
                if (!$columns.ContainsKey($key)) { $columns[$key] = [Collections.Generic.List[double]]::new() }
                $number = 0.0
                if ($cell.t -in @('s','inlineStr','b','e') -or ![double]::TryParse([string]$cell.v, [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$number)) { $nonNumeric++; continue }
                $columns[$key].Add($number)
            }
        }
        $report = [ordered]@{sheet=[string]$sheet.name; rowCount=$rows.Count; nonNumericCells=$nonNumeric; columns=@{}}
        foreach ($key in @($columns.Keys | Sort-Object)) {
            $values=$columns[$key].ToArray()
            $delta=[Collections.Generic.List[double]]::new()
            for ($i=1; $i -lt $values.Count; $i++) { $delta.Add($values[$i]-$values[$i-1]) }
            $identical=0
            foreach ($v in $delta) { if ([Math]::Abs($v) -lt 1e-12) { $identical++ } }
            $equalSteps=0; $run=1; $longestRun=1
            for ($i=1; $i -lt $delta.Count; $i++) {
                if ([Math]::Abs($delta[$i]-$delta[$i-1]) -lt 1e-10) { $equalSteps++; $run++ } else { $longestRun=[Math]::Max($longestRun,$run); $run=1 }
            }
            $longestRun=[Math]::Max($longestRun,$run)
            $report.columns[$key]=[ordered]@{displacement=(Summarize $values); first=$values[0]; last=$values[-1]; firstFive=@($values | Select-Object -First 5); increment=(Summarize $delta.ToArray()); zeroIncrements=$identical}
            $report.columns[$key]['equalAdjacentIncrementFraction']=$equalSteps/($delta.Count-1)
            $report.columns[$key]['longestConstantIncrementRun']=$longestRun
        }
        if ($columns.ContainsKey('A') -and $columns.ContainsKey('B')) {
            $a=$columns['A']; $b=$columns['B']; $n=[Math]::Min($a.Count,$b.Count)
            $ma=($a | Measure-Object -Average).Average; $mb=($b | Measure-Object -Average).Average
            $cross=0.0; $va=0.0; $vb=0.0; $dif=[Collections.Generic.List[double]]::new()
            $crossD=0.0; $vaD=0.0; $vbD=0.0
            $daMean=($a[$n-1]-$a[0])/($n-1); $dbMean=($b[$n-1]-$b[0])/($n-1)
            for ($i=0; $i -lt $n; $i++) {
                $cross+=($a[$i]-$ma)*($b[$i]-$mb); $va+=($a[$i]-$ma)*($a[$i]-$ma); $vb+=($b[$i]-$mb)*($b[$i]-$mb)
                $dif.Add(($b[$i]-$b[0])-($a[$i]-$a[0]))
                if ($i -gt 0) { $da=$a[$i]-$a[$i-1]-$daMean; $db=$b[$i]-$b[$i-1]-$dbMean; $crossD+=$da*$db; $vaD+=$da*$da; $vbD+=$db*$db }
            }
            $report['comparison']=[ordered]@{correlation=($cross/[Math]::Sqrt($va*$vb)); incrementCorrelation=($crossD/[Math]::Sqrt($vaD*$vbD)); zeroedTarget2MinusTarget1=(Summarize $dif.ToArray()); finalDifference=$dif[$n-1]}
        }
        $report | ConvertTo-Json -Depth 8
    }
} finally { $archive.Dispose() }
