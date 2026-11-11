; Instalador offline por usuário. Os caminhos e as cópias compartilhadas
; são gerados por scripts/build_installer.py a partir do pacote validado.
[Setup]
AppId={#InstallerAppId}
AppName=Goal Event Review
AppVersion={#AppVersion}
AppPublisher=Goal Event Review
DefaultDirName={localappdata}\Programs\Goal Event Review
DefaultGroupName=Goal Event Review
PrivilegesRequired=lowest
SetupArchitecture=x64
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputDir={#OutputFolder}
OutputBaseFilename=GoalEventReview-{#AppVersion}-Setup-x64
Compression=none
WizardStyle=modern
DisableProgramGroupPage=yes
UninstallDisplayIcon={app}\Goal Event Review.exe
UninstallDisplayName=Goal Event Review
ExtraDiskSpaceRequired={#InstalledBytes + 536870912}
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "brazilianportuguese"; MessagesFile: "compiler:Languages\BrazilianPortuguese.isl"

[Tasks]
Name: "desktopicon"; Description: "Criar atalho na área de trabalho"; GroupDescription: "Atalhos:"; Flags: unchecked

[Files]
Source: "{#SevenZipBinary}"; DestDir: "{tmp}"; DestName: "goal-review-extractor.exe"; Flags: deleteafterinstall
#include ArchiveEntries
Source: "{#SevenZipLicense}"; DestDir: "{app}"; DestName: "LICENSE.7zip.txt"; Flags: ignoreversion
Source: "{#SevenZipCopying}"; DestDir: "{app}"; DestName: "COPYING.7zip.txt"; Flags: ignoreversion
Source: "{#SevenZipNotice}"; DestDir: "{app}"; DestName: "SOURCE.7zip.txt"; Flags: ignoreversion
Source: "{#TestingGuide}"; DestDir: "{app}"; DestName: "LEIA-ME-TESTE.txt"; Flags: ignoreversion

[Icons]
Name: "{group}\Goal Event Review"; Filename: "{app}\Goal Event Review.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\Goal Event Review"; Filename: "{app}\Goal Event Review.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\Goal Event Review.exe"; Description: "Iniciar Goal Event Review"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
#include UninstallEntries

[Code]
procedure RestoreSharedFiles;
var
  Lines: TArrayOfString;
  I, Separator: Integer;
  Source, Destination, Prefix, RelativeSource, RelativeDestination: String;
begin
  Prefix := AddBackslash(ExpandFileName(ExpandConstant('{app}')));
  if not LoadStringsFromFile(Prefix + 'bundle-aliases.tsv', Lines) then
    RaiseException('Não foi possível ler o manifesto dos componentes compartilhados.');
  for I := 0 to GetArrayLength(Lines) - 1 do begin
    if Lines[I] = '' then Continue;
    Separator := Pos(#9, Lines[I]);
    if Separator = 0 then RaiseException('Manifesto de componentes inválido.');
    RelativeDestination := Copy(Lines[I], 1, Separator - 1);
    RelativeSource := Copy(Lines[I], Separator + 1, Length(Lines[I]));
    StringChangeEx(RelativeDestination, '/', '\', True);
    StringChangeEx(RelativeSource, '/', '\', True);
    Source := ExpandFileName(Prefix + RelativeSource);
    Destination := ExpandFileName(Prefix + RelativeDestination);
    if (CompareText(Copy(Source, 1, Length(Prefix)), Prefix) <> 0) or
      (CompareText(Copy(Destination, 1, Length(Prefix)), Prefix) <> 0) then
      RaiseException('Caminho inválido no manifesto de componentes.');
    ForceDirectories(ExtractFileDir(Destination));
    if not CopyFile(Source, Destination, False) then
      RaiseException('Não foi possível preparar um componente compartilhado: ' + RelativeDestination);
  end;
end;

procedure PrepareComponents;
var
  ArchiveName, PartName, Number, Stage: String;
  Destination, Source: TFileStream;
  I, ExitCode: Integer;
begin
  ArchiveName := ExpandConstant('{tmp}\goal-review.7z');
  Stage := ExpandConstant('{app}');
  WizardForm.StatusLabel.Caption := 'Preparando os componentes incluídos…';
  WizardForm.FilenameLabel.Caption := '';
  Destination := TFileStream.Create(ArchiveName, $FFFF);
  try
    for I := 1 to {#PartCount} do begin
      Number := IntToStr(I);
      while Length(Number) < 3 do Number := '0' + Number;
      PartName := ArchiveName + '.' + Number;
      Source := TFileStream.Create(PartName, $20);
      try
        Destination.CopyFrom(Source, 0, 1048576);
      finally
        Source.Free;
      end;
      DeleteFile(PartName);
    end;
  finally
    Destination.Free;
  end;
  ForceDirectories(Stage);
  WizardForm.StatusLabel.Caption := 'Descompactando CPU, NVIDIA e AMD. Aguarde…';
  if not Exec(ExpandConstant('{tmp}\goal-review-extractor.exe'),
    'x -y -mmt=1 -bso0 -bsp0 -o"' + Stage + '" "' + ArchiveName + '"',
    '', SW_HIDE, ewWaitUntilTerminated, ExitCode) then
    RaiseException('Não foi possível iniciar a preparação dos componentes.');
  if ExitCode <> 0 then
    RaiseException('Não foi possível descompactar os componentes. Código: ' + IntToStr(ExitCode));
  DeleteFile(ArchiveName);
  WizardForm.StatusLabel.Caption := 'Finalizando os componentes compartilhados…';
  RestoreSharedFiles;
end;
