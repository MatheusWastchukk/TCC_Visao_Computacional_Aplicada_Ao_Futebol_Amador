using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text;

static class ArchivePreparation {
    static void Resource(string name, string target) {
        using (var input = Assembly.GetExecutingAssembly().GetManifestResourceStream(name))
        using (var output = File.Create(Bundle.Extended(target))) {
            if (input == null) throw new IOException("O extrator incluído está ausente.");
            input.CopyTo(output, 1024 * 1024);
        }
    }
    public static void Extract(Bundle bundle, string staging, Action check, Action<string, double> status) {
        if (new DriveInfo(Path.GetPathRoot(bundle.Root)).AvailableFreeSpace < bundle.Expanded + bundle.Length + 512L * 1024 * 1024)
            throw new IOException("Espaço insuficiente. Libere pelo menos " + Math.Ceiling((bundle.Expanded + bundle.Length) / 1073741824.0).ToString() + " GB para a preparação inicial.");
        if (Directory.Exists(staging)) Directory.Delete(Bundle.Extended(bundle.Safe(staging)), true);
        Directory.CreateDirectory(staging);
        string archive = Path.Combine(staging, ".payload.7z");
        string helper = Path.Combine(staging, ".extractor.exe");
        Resource("SevenZip", helper);
        Resource("SevenZipLicense", Path.Combine(staging, "LICENSE.7zip.txt"));
        Resource("SevenZipCopying", Path.Combine(staging, "COPYING.7zip.txt"));
        File.WriteAllText(Path.Combine(staging, "SOURCE.7zip.txt"), "7-Zip 26.06, Copyright Igor Pavlov.\nSource and license: https://github.com/ip7z/7zip/tree/26.06\nhttps://www.7-zip.org/\n");
        using (var input = new SliceStream(bundle.Self, bundle.Offset, bundle.Length))
        using (var output = File.Create(archive)) {
            var buffer = new byte[1024 * 1024]; int read; long done = 0; var watch = Stopwatch.StartNew();
            while ((read = input.Read(buffer, 0, buffer.Length)) > 0) {
                check(); output.Write(buffer, 0, read); done += read;
                if (watch.ElapsedMilliseconds > 200) { status("Preparando o arquivo de componentes…", 0.15 + 0.15 * done / bundle.Length); watch.Restart(); }
            }
        }
        status("Descompactando CPU, NVIDIA e AMD…\nIsso é necessário apenas na primeira abertura.", -1);
        var start = new ProcessStartInfo(helper, "x -y -mmt=1 -bso0 -bsp0 -bse1 -o" + Bundle.Quote(staging) + " " + Bundle.Quote(archive));
        start.UseShellExecute = false; start.CreateNoWindow = true;
        start.RedirectStandardOutput = true; start.RedirectStandardError = true;
        using (var process = Process.Start(start)) {
            var stdout = process.StandardOutput.ReadToEndAsync(); var stderr = process.StandardError.ReadToEndAsync();
            try { while (!process.WaitForExit(200)) check(); }
            catch { if (!process.HasExited) { process.Kill(); process.WaitForExit(); } throw; }
            stdout.Wait(); stderr.Wait();
            if (process.ExitCode != 0) throw new IOException("O extrator não concluiu a preparação: " + stderr.Result);
        }
        check();
        File.Delete(archive); File.Delete(helper);
        string aliases = Path.Combine(staging, "bundle-aliases.tsv");
        if (File.Exists(aliases)) {
            status("Finalizando os componentes compartilhados…", 0.95);
            string prefix = Path.GetFullPath(staging) + Path.DirectorySeparatorChar;
            foreach (string line in File.ReadLines(aliases)) {
                check(); if (line.Length == 0) continue;
                string[] pair = line.Split('\t');
                if (pair.Length != 2) throw new IOException("Manifesto de componentes inválido.");
                string target = Path.GetFullPath(Path.Combine(staging, pair[0]));
                string original = Path.GetFullPath(Path.Combine(staging, pair[1]));
                if (!target.StartsWith(prefix, StringComparison.OrdinalIgnoreCase) || !original.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                    throw new IOException("O manifesto contém um caminho inválido.");
                Directory.CreateDirectory(Bundle.Extended(Path.GetDirectoryName(target)));
                File.Copy(Bundle.Extended(original), Bundle.Extended(target), true);
            }
        }
        status("Finalizando a preparação…", 0.99);
    }
}
