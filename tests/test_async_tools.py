"""Tests for async filesystem tools."""

import pytest
from deepagents.middleware.filesystem import _get_async_filesystem_tools, FilesystemState
from deepagents.backends import StateBackend
from langchain.tools import ToolRuntime


@pytest.fixture
def runtime():
    """Create a test runtime."""
    state = FilesystemState(messages=[], files={})
    return ToolRuntime(
        state=state,
        context=None,
        tool_call_id="test_id",
        store=None,
        stream_writer=lambda _: None,
        config={}
    )


@pytest.mark.asyncio
async def test_async_ls_tool(runtime):
    """Test async ls tool."""
    backend = StateBackend(runtime)
    tools = _get_async_filesystem_tools(backend)
    
    als_tool = next(t for t in tools if t.name == "als")
    
    # Test listing root directory (empty)
    result = await als_tool.ainvoke({"runtime": runtime, "path": "/"})
    assert result == []


@pytest.mark.asyncio
async def test_async_write_and_read_tools(runtime):
    """Test async write and read tools."""
    backend = StateBackend(runtime)
    tools = _get_async_filesystem_tools(backend)
    
    awrite_tool = next(t for t in tools if t.name == "awrite_file")
    aread_tool = next(t for t in tools if t.name == "aread_file")
    als_tool = next(t for t in tools if t.name == "als")
    
    # Write a file
    write_result = await awrite_tool.ainvoke({
        "file_path": "/test.txt",
        "content": "Hello, world!",
        "runtime": runtime,
    })
    
    # Check write result - it should be a Command with update
    from langgraph.types import Command
    assert isinstance(write_result, Command)
    assert "messages" in write_result.update
    assert "files" in write_result.update
    
    # Manually update the state (in real execution, this would be done by LangGraph)
    runtime.state["files"].update(write_result.update["files"])
    
    # Read the file
    read_result = await aread_tool.ainvoke({
        "file_path": "/test.txt",
        "runtime": runtime,
    })
    
    assert "Hello, world!" in read_result
    
    # List files
    ls_result = await als_tool.ainvoke({"runtime": runtime, "path": "/"})
    assert "/test.txt" in ls_result


@pytest.mark.asyncio
async def test_async_edit_tool(runtime):
    """Test async edit tool."""
    backend = StateBackend(runtime)
    tools = _get_async_filesystem_tools(backend)
    
    awrite_tool = next(t for t in tools if t.name == "awrite_file")
    aedit_tool = next(t for t in tools if t.name == "aedit_file")
    aread_tool = next(t for t in tools if t.name == "aread_file")
    
    from langgraph.types import Command
    
    # Write a file
    write_result = await awrite_tool.ainvoke({
        "file_path": "/test.txt",
        "content": "Hello, world!",
        "runtime": runtime,
    })
    runtime.state["files"].update(write_result.update["files"])
    
    # Edit the file
    edit_result = await aedit_tool.ainvoke({
        "file_path": "/test.txt",
        "old_string": "world",
        "new_string": "async",
        "runtime": runtime,
    })
    
    # Check edit result
    assert isinstance(edit_result, Command)
    assert "messages" in edit_result.update
    runtime.state["files"].update(edit_result.update["files"])
    
    # Read the edited file
    read_result = await aread_tool.ainvoke({
        "file_path": "/test.txt",
        "runtime": runtime,
    })
    
    assert "Hello, async!" in read_result


@pytest.mark.asyncio
async def test_async_glob_tool(runtime):
    """Test async glob tool."""
    backend = StateBackend(runtime)
    tools = _get_async_filesystem_tools(backend)
    
    awrite_tool = next(t for t in tools if t.name == "awrite_file")
    aglob_tool = next(t for t in tools if t.name == "aglob")
    
    # Write some files
    result1 = await awrite_tool.ainvoke({
        "file_path": "/test1.txt",
        "content": "Content 1",
        "runtime": runtime,
    })
    runtime.state["files"].update(result1.update["files"])
    
    result2 = await awrite_tool.ainvoke({
        "file_path": "/test2.txt",
        "content": "Content 2",
        "runtime": runtime,
    })
    runtime.state["files"].update(result2.update["files"])
    
    result3 = await awrite_tool.ainvoke({
        "file_path": "/test.py",
        "content": "print('hello')",
        "runtime": runtime,
    })
    runtime.state["files"].update(result3.update["files"])
    
    # Search for .txt files
    glob_result = await aglob_tool.ainvoke({
        "pattern": "*.txt",
        "runtime": runtime,
        "path": "/",
    })
    
    assert len(glob_result) == 2
    assert "/test1.txt" in glob_result
    assert "/test2.txt" in glob_result
    assert "/test.py" not in glob_result


@pytest.mark.asyncio
async def test_async_grep_tool(runtime):
    """Test async grep tool."""
    backend = StateBackend(runtime)
    tools = _get_async_filesystem_tools(backend)
    
    awrite_tool = next(t for t in tools if t.name == "awrite_file")
    agrep_tool = next(t for t in tools if t.name == "agrep")
    
    # Write files with different content
    result1 = await awrite_tool.ainvoke({
        "file_path": "/test1.txt",
        "content": "This file contains the word hello",
        "runtime": runtime,
    })
    runtime.state["files"].update(result1.update["files"])
    
    result2 = await awrite_tool.ainvoke({
        "file_path": "/test2.txt",
        "content": "This file contains the word goodbye",
        "runtime": runtime,
    })
    runtime.state["files"].update(result2.update["files"])
    
    result3 = await awrite_tool.ainvoke({
        "file_path": "/test3.txt",
        "content": "This file contains the word hello too",
        "runtime": runtime,
    })
    runtime.state["files"].update(result3.update["files"])
    
    # Search for "hello"
    grep_result = await agrep_tool.ainvoke({
        "pattern": "hello",
        "runtime": runtime,
        "output_mode": "files_with_matches",
    })
    
    assert "/test1.txt" in grep_result
    assert "/test3.txt" in grep_result
    assert "/test2.txt" not in grep_result


@pytest.mark.asyncio
async def test_async_tools_work_with_filesystem_backend():
    """Test async tools with FilesystemBackend."""
    import tempfile
    import os
    from deepagents.backends.filesystem import FilesystemBackend
    
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = FilesystemBackend(root_dir=tmpdir, virtual_mode=True)
        
        # Create a test file
        test_file = os.path.join(tmpdir, "test.txt")
        with open(test_file, "w") as f:
            f.write("Hello from filesystem!")
        
        tools = _get_async_filesystem_tools(backend)
        
        # Create a minimal runtime
        state = FilesystemState(messages=[], files={})
        runtime = ToolRuntime(
            state=state,
            context=None,
            tool_call_id="test_id",
            store=None,
            stream_writer=lambda _: None,
            config={}
        )
        
        als_tool = next(t for t in tools if t.name == "als")
        aread_tool = next(t for t in tools if t.name == "aread_file")
        
        # Test ls
        ls_result = await als_tool.ainvoke({"runtime": runtime, "path": "/"})
        assert "/test.txt" in ls_result
        
        # Test read
        read_result = await aread_tool.ainvoke({
            "file_path": "/test.txt",
            "runtime": runtime,
        })
        assert "Hello from filesystem!" in read_result
