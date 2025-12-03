# Run
#
#   nix develop
#
# to run development shell.
{
  description = "Papaya Spam Eater devshell";
  inputs = {
    nixpkgs.url = "nixpkgs/nixos-25.11";
    # or for unstable
    # nixpkgs.url = "nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = {
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = nixpkgs.legacyPackages.${system};
      pythonPatched = pkgs.python3.withPackages (pp:
        with pp; [
          # Runtime dependencies
          watchdog
          beautifulsoup4
          lxml
          scikit-learn
          pyyaml
          typer

          # Development dependencies
          pytest
          pytest-cov
          pytest-asyncio
          pytest-timeout
          hypothesis
          freezegun
          faker
          ruff
          mypy
          types-pyyaml
        ]);
      buildInputs = [
        pythonPatched
        pkgs.uv
        pkgs.just
      ];
    in {
      devShells.default = pkgs.mkShell {
        inherit buildInputs;
      };
    });
}
