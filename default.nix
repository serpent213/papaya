{
  lib ? (import <nixpkgs> {}).lib,
  python3Packages ? (import <nixpkgs> {}).python3Packages,
}:
python3Packages.buildPythonApplication {
  pname = "papaya";
  version = "0.1.0";
  format = "pyproject";

  src = ./.;

  nativeBuildInputs = with python3Packages; [
    setuptools
  ];

  propagatedBuildInputs = with python3Packages; [
    watchdog
    beautifulsoup4
    lxml
    scikit-learn
    pyyaml
    typer
  ];

  checkInputs = with python3Packages; [
    pytestCheckHook
    pytest-cov
    pytest-asyncio
    pytest-timeout
    hypothesis
    freezegun
    faker
  ];

  meta = {
    description = "Personal/small-scale maildir spam classification daemon";
    longDescription = ''
      A machine learning daemon that watches Maildir folders and classifies spam locally
    '';
    homepage = "https://github.com/serpent213/papaya";
    platforms = lib.platforms.all;
    mainProgram = "papaya";
    license = lib.licenses.bsd2;
    maintainers = with lib.maintainers; [serpent213];
  };
}
