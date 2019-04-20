devenv .\build\Vision.sln /build
xcopy .\bin\Release\sgbm.dll .\sgbm.pyd /y
xcopy .\bin\Release\pyelas.dll .\pyelas.pyd /y
pause